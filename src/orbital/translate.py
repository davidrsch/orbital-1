"""Translate a pipeline into an Ibis expression."""

import logging
import typing

import ibis

from .ast import ParsedPipeline
from .translation._registry import TRANSLATORS  # noqa: F401
from .translation.optimizer import Optimizer
from .translation.options import TranslationOptions
from .translation.translator import Translator
from .translation.variables import GraphVariables

log = logging.getLogger(__name__)

# This is primarily for development purposes.
# It's disabled by default because it implies
# a significant cost of executing queries on each step.
LOG_DATA = False
LOG_SQL = False


class ResultsProjection:
    """Projection of the results of the pipeline.

    This class is used to select the columns to be returned
    from the pipeline. It can be used to select specific
    columns to include in the final result set.

    It can also be used to skip the select step of columns
    from the pipeline.

    You can use the `omit` method to skip the projection
    step entirely.
    """

    def __init__(self, select: typing.Optional[list[str]] = None) -> None:
        """
        :param select: A list of additional columns to be selected from the pipeline.
        """
        self._select = select or []
        self._omit = False

    @classmethod
    def omit(cls) -> "ResultsProjection":
        """Create a projection that skips projection phase entirely."""
        projection = cls()
        projection._omit = True
        return projection

    def _expand(self, results: typing.Iterable[str]) -> typing.Optional[list[str]]:
        if self._omit:
            return None

        def _emit_projection() -> typing.Generator[str, None, None]:
            yield from results
            for item in self._select:
                yield item

        return list(_emit_projection())


def translate(
    table: ibis.Table,
    pipeline: ParsedPipeline,
    projection: typing.Optional[ResultsProjection] = None,
    *,
    allow_text_tensors: bool = False,
) -> ibis.Table:
    """Translate a pipeline into an Ibis expression.

    Converts the provided ``orbital.ast.ParsedPipeline`` into an Ibis
    expression that reproduces the same transformations. The expression can be
    executed directly or further composed before exporting to SQL.

    :param table: Source ibis table used as the translation input.
    :param pipeline: Parsed pipeline to be translated.
    :param projection: Optional result projection helper.
    :param allow_text_tensors: When ``False`` (default) skip ONNX casts that
        promote numeric/bool tensors to strings solely to coexist with
        passthrough text features. Set to ``True`` to preserve the casts
        exactly as exported.
    """
    if projection is None:
        projection = ResultsProjection()
    optimizer = Optimizer(enabled=True)
    options = TranslationOptions(allow_text_tensors=allow_text_tensors)
    features = {colname: table[colname] for colname in table.columns}
    variables = GraphVariables(features, pipeline._model.graph)
    nodes = list(pipeline._model.graph.node)
    for node in nodes:
        op_type = node.op_type
        if op_type not in TRANSLATORS:
            raise NotImplementedError(f"Translation for {op_type} not implemented")
        if op_type == "Attention" and node.domain != "com.microsoft":
            raise NotImplementedError(
                "Attention: only the com.microsoft contrib op is supported. "
                "The standard ai.onnx.Attention op (ONNX opset 23+) uses different "
                "semantics and is not supported."
            )
        if op_type == "SkipLayerNormalization" and node.domain != "com.microsoft":
            raise NotImplementedError(
                "SkipLayerNormalization: only the com.microsoft contrib op is supported."
            )
        translator = TRANSLATORS[op_type](table, node, variables, optimizer, options)  # type: ignore[abstract]
        _log_debug_start(translator, variables)
        translator.process()
        table = translator.mutated_table  # Translator might return a new table.
        _log_debug_end(translator, variables)
    return _projection_results(table, variables, projection)


def _projection_results(
    table: ibis.Table,
    variables: GraphVariables,
    projection: typing.Optional[ResultsProjection] = None,
) -> ibis.Table:
    if projection is None:
        projection = ResultsProjection()
    # As we pop out the variables as we use them
    # the remaining ones are the values resulting from all
    # graph branches.
    final_projections: dict[str, typing.Any] = {}
    for key, value in variables.remaining().items():
        if isinstance(value, dict):
            for field in value:
                colkey = key + "." + field
                colvalue = value[field]
                if isinstance(colvalue, ibis.expr.types.StructColumn):
                    raise NotImplementedError(f"StructColumn not supported: {colvalue}")
                final_projections[colkey] = colvalue
        else:
            final_projections[key] = value
    query = table.mutate(**final_projections)
    selection = projection._expand(final_projections.keys())
    if selection is not None:
        query = query.select(*selection)
    return query


def _log_debug_start(translator: Translator, variables: GraphVariables) -> None:
    debug_inputs = {}
    node = translator._node
    for inp in translator._inputs:
        value: typing.Any = None
        if (feature_value := translator._variables.peek_variable(inp)) is not None:
            value = type(feature_value)
        elif (
            initializer := translator._variables.get_initializer_value(inp)
        ) is not None:
            value = initializer
        else:
            raise ValueError(
                f"Unknown input: {inp} for node {node.name}({translator.__class__.__name__})"
            )
        debug_inputs[inp] = value
    log.debug(
        f"Node: {node.name}, Op: {node.op_type}, Attributes: {translator._attributes}, Inputs: {debug_inputs}"
    )
    if LOG_DATA:
        print("Input Data", flush=True)
        print(
            _projection_results(translator.mutated_table, variables).execute(),
            flush=True,
        )
        print("", flush=True)


def _log_debug_end(translator: Translator, variables: GraphVariables) -> None:
    variables = translator._variables
    output_vars = {
        name: type(variables.peek_variable(name)) for name in translator.outputs
    }
    log.debug(
        f"\tOutput: {output_vars} TOTAL: {variables.nested_len()}/{len(variables)}"
    )

    if LOG_DATA:
        print("\tOutput Data", flush=True)
        print(
            _projection_results(translator.mutated_table, variables).execute(),
            flush=True,
        )
        print("", flush=True)
    if LOG_SQL:
        print("\tSQL Expressions", flush=True)
        print(
            ibis.duckdb.connect().compile(
                (_projection_results(translator.mutated_table, variables))
            ),
            flush=True,
        )
