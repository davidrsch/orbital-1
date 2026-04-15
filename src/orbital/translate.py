"""Translate a pipeline into an Ibis expression."""

import logging
import typing

import ibis

from .ast import ParsedPipeline
from .translation.optimizer import Optimizer
from .translation.options import TranslationOptions
from .translation.steps.abs import AbsTranslator
from .translation.steps.add import AddTranslator
from .translation.steps.argmax import ArgMaxTranslator
from .translation.steps.arrayfeatureextractor import ArrayFeatureExtractorTranslator
from .translation.steps.avgpool import AveragePoolTranslator
from .translation.steps.batchnorm import BatchNormalizationTranslator
from .translation.steps.cast import CastTranslator
from .translation.steps.castlike import CastLikeTranslator
from .translation.steps.celu import CeluTranslator
from .translation.steps.conv import ConvTranslator
from .translation.steps.convtranspose import ConvTransposeTranslator
from .translation.steps.gru import GRUTranslator
from .translation.steps.lstm import LSTMTranslator
from .translation.steps.clip import ClipTranslator
from .translation.steps.concat import ConcatTranslator
from .translation.steps.featurevectorizer import FeatureVectorizerTranslator
from .translation.steps.div import DivTranslator
from .translation.steps.dropout import DropoutTranslator
from .translation.steps.elu import EluTranslator
from .translation.steps.ceil import CeilTranslator
from .translation.steps.erf import ErfTranslator
from .translation.steps.exp import ExpTranslator
from .translation.steps.flatten import FlattenTranslator
from .translation.steps.floor import FloorTranslator
from .translation.steps.log import LogTranslator
from .translation.steps.gather import GatherTranslator
from .translation.steps.gelu import GeluTranslator
from .translation.steps.gemm import GemmTranslator
from .translation.steps.globalavgpool import GlobalAveragePoolTranslator
from .translation.steps.globalmaxpool import GlobalMaxPoolTranslator
from .translation.steps.groupnorm import GroupNormalizationTranslator
from .translation.steps.hardsigmoid import HardSigmoidTranslator
from .translation.steps.hardswish import HardSwishTranslator
from .translation.steps.hardtanh import HardTanhTranslator
from .translation.steps.identity import IdentityTranslator
from .translation.steps.imputer import ImputerTranslator
from .translation.steps.isinf import IsInfTranslator
from .translation.steps.isnan import IsNaNTranslator
from .translation.steps.instancenorm import InstanceNormalizationTranslator
from .translation.steps.labelencoder import LabelEncoderTranslator
from .translation.steps.layernorm import LayerNormalizationTranslator
from .translation.steps.leakyrelu import LeakyReluTranslator
from .translation.steps.localresponsenormalization import LRNTranslator
from .translation.steps.logsigmoid import LogSigmoidTranslator
from .translation.steps.logsoftmax import LogSoftmaxTranslator
from .translation.steps.lpnormalization import LpNormalizationTranslator
from .translation.steps.linearclass import LinearClassifierTranslator
from .translation.steps.linearreg import LinearRegressorTranslator
from .translation.steps.matmul import MatMulTranslator
from .translation.steps.maxpool import MaxPoolTranslator
from .translation.steps.mish import MishTranslator
from .translation.steps.mod import ModTranslator
from .translation.steps.meanvariancenorm import MeanVarianceNormalizationTranslator
from .translation.steps.attention import AttentionTranslator
from .translation.steps.multiheadattention import MultiHeadAttentionTranslator
from .translation.steps.skip_layer_norm import SkipLayerNormalizationTranslator
from .translation.steps.mul import MulTranslator
from .translation.steps.neg import NegTranslator
from .translation.steps.onehotencoder import OneHotEncoderTranslator
from .translation.steps.pad import PadTranslator
from .translation.steps.pow import PowTranslator
from .translation.steps.prelu import PReluTranslator
from .translation.steps.reducemax import ReduceMaxTranslator
from .translation.steps.reducemean import ReduceMeanTranslator
from .translation.steps.reducemin import ReduceMinTranslator
from .translation.steps.reducesum import ReduceSumTranslator
from .translation.steps.reducel1 import ReduceL1Translator
from .translation.steps.reducel2 import ReduceL2Translator
from .translation.steps.reducelogsum import ReduceLogSumTranslator
from .translation.steps.reducelogsumexp import ReduceLogSumExpTranslator
from .translation.steps.reducesumsquare import ReduceSumSquareTranslator
from .translation.steps.variadicsum import VariadicSumTranslator
from .translation.steps.variadicmax import VariadicMaxTranslator
from .translation.steps.variadicmin import VariadicMinTranslator
from .translation.steps.variadicmean import VariadicMeanTranslator
from .translation.steps.reduceprod import ReduceProdTranslator
from .translation.steps.relu import ReluTranslator
from .translation.steps.reshape import ReshapeTranslator
from .translation.steps.round_ import RoundTranslator
from .translation.steps.rmsnorm import RMSNormalizationTranslator
from .translation.steps.rnn import RNNTranslator
from .translation.steps.scaler import ScalerTranslator
from .translation.steps.sign import SignTranslator
from .translation.steps.shrink import ShrinkTranslator
from .translation.steps.scatterelements import ScatterElementsTranslator
from .translation.steps.shape import ShapeTranslator
from .translation.steps.split import SplitTranslator
from .translation.steps.selu import SeluTranslator
from .translation.steps.sigmoid import SigmoidTranslator
from .translation.steps.softmax import SoftmaxTranslator
from .translation.steps.softplus import SoftplusTranslator
from .translation.steps.softsign import SoftsignTranslator
from .translation.steps.sqrt import SqrtTranslator
from .translation.steps.squeeze import SqueezeTranslator
from .translation.steps.unsqueeze import UnsqueezeTranslator
from .translation.steps.sub import SubTranslator
from .translation.steps.swish import SwishTranslator
from .translation.steps.tanh import TanhTranslator
from .translation.steps.thresholdedrelu import ThresholdedReluTranslator
from .translation.steps.slice import SliceTranslator
from .translation.steps.tile import TileTranslator
from .translation.steps.transpose import TransposeTranslator
from .translation.steps.trees import (
    TreeEnsembleClassifierTranslator,
    TreeEnsembleRegressorTranslator,
)
from .translation.steps.where import WhereTranslator
from .translation.steps.zipmap import ZipMapTranslator
from .translation.translator import Translator
from .translation.variables import GraphVariables

# This is a mapping of ONNX operations to their respective translators
# It could be implemented via some form of autodiscovery and
# registration, but explicit mapping avoids effects at a distance and
# makes it easier to understand the translation process.
TRANSLATORS: dict[str, type[Translator]] = {
    "Abs": AbsTranslator,
    "AveragePool": AveragePoolTranslator,
    "BatchNormalization": BatchNormalizationTranslator,
    "Cast": CastTranslator,
    "CastLike": CastLikeTranslator,
    "Ceil": CeilTranslator,
    "Celu": CeluTranslator,
    "Clip": ClipTranslator,
    "Concat": ConcatTranslator,
    "Conv": ConvTranslator,
    "ConvTranspose": ConvTransposeTranslator,
    "GRU": GRUTranslator,
    "LSTM": LSTMTranslator,
    "Dropout": DropoutTranslator,
    "Elu": EluTranslator,
    "Erf": ErfTranslator,
    "Exp": ExpTranslator,
    "Log": LogTranslator,
    "FeatureVectorizer": FeatureVectorizerTranslator,
    "Gelu": GeluTranslator,
    "GlobalAveragePool": GlobalAveragePoolTranslator,
    "GlobalMaxPool": GlobalMaxPoolTranslator,
    "GroupNormalization": GroupNormalizationTranslator,
    "HardSigmoid": HardSigmoidTranslator,
    "HardSwish": HardSwishTranslator,
    "HardTanh": HardTanhTranslator,
    "InstanceNormalization": InstanceNormalizationTranslator,
    "LayerNormalization": LayerNormalizationTranslator,
    "LpNormalization": LpNormalizationTranslator,
    "LRN": LRNTranslator,
    "LeakyRelu": LeakyReluTranslator,
    "LogSigmoid": LogSigmoidTranslator,
    "Sub": SubTranslator,
    "MatMul": MatMulTranslator,
    "Add": AddTranslator,
    "Div": DivTranslator,
    "Mul": MulTranslator,
    "Mish": MishTranslator,
    "Mod": ModTranslator,
    "MeanVarianceNormalization": MeanVarianceNormalizationTranslator,
    "Attention": AttentionTranslator,
    "MultiHeadAttention": MultiHeadAttentionTranslator,
    "SkipLayerNormalization": SkipLayerNormalizationTranslator,
    "Neg": NegTranslator,
    "Pad": PadTranslator,
    "Pow": PowTranslator,
    "PRelu": PReluTranslator,
    "ReduceL1": ReduceL1Translator,
    "ReduceL2": ReduceL2Translator,
    "ReduceLogSum": ReduceLogSumTranslator,
    "ReduceLogSumExp": ReduceLogSumExpTranslator,
    "ReduceSumSquare": ReduceSumSquareTranslator,
    "Sum": VariadicSumTranslator,
    "Max": VariadicMaxTranslator,
    "Min": VariadicMinTranslator,
    "Mean": VariadicMeanTranslator,
    "ReduceMax": ReduceMaxTranslator,
    "ReduceMean": ReduceMeanTranslator,
    "ReduceMin": ReduceMinTranslator,
    "ReduceSum": ReduceSumTranslator,
    "ReduceProd": ReduceProdTranslator,
    "RNN": RNNTranslator,
    "Sign": SignTranslator,
    "Shrink": ShrinkTranslator,
    "ScatterElements": ScatterElementsTranslator,
    "Shape": ShapeTranslator,
    "Slice": SliceTranslator,
    "Split": SplitTranslator,
    "Tile": TileTranslator,
    "Flatten": FlattenTranslator,
    "Floor": FloorTranslator,
    "LogSoftmax": LogSoftmaxTranslator,
    "MaxPool": MaxPoolTranslator,
    "Relu": ReluTranslator,
    "Reshape": ReshapeTranslator,
    "Round": RoundTranslator,
    "RMSNormalization": RMSNormalizationTranslator,
    "Transpose": TransposeTranslator,
    "Scaler": ScalerTranslator,
    "Selu": SeluTranslator,
    "Sigmoid": SigmoidTranslator,
    "Softmax": SoftmaxTranslator,
    "Softplus": SoftplusTranslator,
    "Softsign": SoftsignTranslator,
    "Swish": SwishTranslator,
    "ThresholdedRelu": ThresholdedReluTranslator,
    "Gather": GatherTranslator,
    "Gemm": GemmTranslator,
    "ArrayFeatureExtractor": ArrayFeatureExtractorTranslator,
    "Identity": IdentityTranslator,
    "Imputer": ImputerTranslator,
    "IsInf": IsInfTranslator,
    "IsNaN": IsNaNTranslator,
    "LabelEncoder": LabelEncoderTranslator,
    "OneHotEncoder": OneHotEncoderTranslator,
    "Where": WhereTranslator,
    "ZipMap": ZipMapTranslator,
    "ArgMax": ArgMaxTranslator,
    "Sqrt": SqrtTranslator,
    "Squeeze": SqueezeTranslator,
    "Tanh": TanhTranslator,
    "Unsqueeze": UnsqueezeTranslator,
    "TreeEnsembleClassifier": TreeEnsembleClassifierTranslator,
    "TreeEnsembleRegressor": TreeEnsembleRegressorTranslator,
    "LinearRegressor": LinearRegressorTranslator,
    "LinearClassifier": LinearClassifierTranslator,
}

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
