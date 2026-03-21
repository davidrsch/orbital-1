# Supported ONNX Operators

Orbital translates scikit-learn pipelines to SQL by converting the ONNX graph that
[skl2onnx](https://onnx.ai/sklearn-onnx/) emits node-by-node into
[ibis](https://ibis-project.org/) expressions.

The table below lists every ONNX op-type that is currently registered in
`orbital.translate.TRANSLATORS`, the typical source model(s) that emit it, and
any known limitations.

## Standard ONNX operators

| Op-type             | Typical source                                               | Notes                                                           |
| ------------------- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| `Add`               | `StandardScaler` (bias), MLP bias term, residual connections | Supports both constant-offset and variable-to-variable addition |
| `ArgMax`            | multiclass classifiers                                       | Returns index of max value                                      |
| `Cast` / `CastLike` | type normalisation steps                                     |                                                                 |
| `Clip`              | `relu6`, quantisation-aware models                           | Supports opset < 11 (attributes) and ≥ 11 (initializer inputs)  |
| `Concat`            | `ColumnTransformer`, `FeatureUnion`                          | axis=1 (column concat) only                                     |
| `Div`               | `StandardScaler` (scale)                                     |                                                                 |
| `Flatten`           | PyTorch MLP preprocessing                                    | axis=1 pass-through                                             |
| `Gather`            | category indexing                                            |                                                                 |
| `Gemm`              | `MLPRegressor`, `MLPClassifier`, `LinearRegression`          | Full `alpha·A@B + beta·C`, transA/transB                        |
| `Identity`          | no-op pass-through                                           |                                                                 |
| `LogSoftmax`        | PyTorch log-probability output heads                         | Numerically stable log-sum-exp                                  |
| `MatMul`            | general matrix multiply                                      |                                                                 |
| `Mul`               | element-wise multiply                                        |                                                                 |
| `Relu`              | MLP hidden layers (`relu` activation)                        |                                                                 |
| `Reshape`           | see note                                                     | No-op when shape is compatible; raises for unsupported reshapes |
| `Sigmoid`           | `LogisticRegression`, binary MLP output                      |                                                                 |
| `Softmax`           | multiclass MLP output                                        | axis=-1 / axis=1                                                |
| `Sub`               | `StandardScaler` (mean)                                      |                                                                 |
| `Tanh`              | MLP hidden layers (`tanh` activation)                        |                                                                 |
| `Transpose`         | PyTorch weight permutation                                   | 2-D pass-through                                                |
| `Where`             | conditional expressions                                      |                                                                 |

## ONNX ML operators (ai.onnx.ml)

| Op-type                  | Typical source                                            | Notes                         |
| ------------------------ | --------------------------------------------------------- | ----------------------------- |
| `ArrayFeatureExtractor`  | `ColumnTransformer`                                       |                               |
| `FeatureVectorizer`      | `ColumnTransformer`                                       | column concat                 |
| `Imputer`                | `SimpleImputer`                                           | mean / median / most-frequent |
| `LabelEncoder`           | `OrdinalEncoder`, label maps                              |                               |
| `LinearClassifier`       | `LogisticRegression`, `SVM`                               |                               |
| `LinearRegressor`        | `LinearRegression`, ridge, lasso                          |                               |
| `OneHotEncoder`          | `OneHotEncoder`                                           |                               |
| `Scaler`                 | `StandardScaler`, `MinMaxScaler`                          |                               |
| `TreeEnsembleClassifier` | `RandomForest`, `GradientBoosting`, `XGBoost`, `LightGBM` |                               |
| `TreeEnsembleRegressor`  | `RandomForest`, `GradientBoosting`                        |                               |
| `ZipMap`                 | classifier post-processing                                | maps scores to class labels   |

## Limitations and recommendations

### SQL expression length

Deep MLP networks generate very long SQL expressions — one nested expression
per neuron per layer. Some databases enforce statement-length or query-plan limits.
For networks with more than ~3 hidden layers and ~100 units per layer, consider
materialising intermediate layer outputs into temporary tables:

```python
# Instead of one huge query, materialise each hidden layer:
layer1_sql = orbital.export_sql("data", layer1_pipeline, dialect="duckdb")
conn.execute(f"CREATE TEMP TABLE layer1 AS ({layer1_sql})")
output_sql = orbital.export_sql("layer1", output_pipeline, dialect="duckdb")
```

### Supported ONNX opsets

Orbital is tested against skl2onnx with the default opset (currently opset 17).
Older models exported with opset < 11 may differ in how `Clip` min/max bounds
are encoded (attributes vs inputs); both variants are handled.

### Unsupported patterns

The following ONNX patterns are not currently translatable to SQL and will raise
a `KeyError` or `NotImplementedError`:

- **Attention / self-attention** (`Attention`, `MultiHeadAttention`)
- **Recurrent layers** (`LSTM`, `GRU`, `RNN`)
- **Dynamic control flow** (`Loop`, `If`, `Scan`)
- **Batch normalisation at inference** (`BatchNormalization`) — planned

### Non-feedforward (DAG) models

Orbital processes ONNX graphs that form a directed acyclic graph (DAG), not just
linear chains. Each intermediate node output is a composable ibis expression;
feeding it into two downstream nodes is free. Residual / skip connections
(via `Add`) are supported as of this release.

### PyTorch models

Orbital officially supports pipelines exported via
[skl2onnx](https://onnx.ai/sklearn-onnx/). PyTorch models exported via
`torch.onnx.export()` often use `Gemm`, `Relu`, `Tanh`, `Flatten`, and
`Transpose` — all of which are now registered. Simple PyTorch
`Sequential(Linear, ReLU, Linear, ...)` MLP models should translate correctly.
More complex architectures (attention, convolutions, residual blocks beyond Add)
are not yet supported.
