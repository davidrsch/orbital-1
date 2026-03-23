# Supported ONNX Operators

Orbital translates scikit-learn pipelines to SQL by converting the ONNX graph that
[skl2onnx](https://onnx.ai/sklearn-onnx/) emits node-by-node into
[ibis](https://ibis-project.org/) expressions.

The table below lists every ONNX op-type that is currently registered in
`orbital.translate.TRANSLATORS`, the typical source model(s) that emit it, and
any known limitations.

## Standard ONNX operators

| Op-type                 | Typical source                                               | Notes                                                                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `Abs`                   | Element-wise absolute value; used in norm/activation exprs   | `\|x\|`                                                                                                       |
| `Add`                   | `StandardScaler` (bias), MLP bias term, residual connections | Supports both constant-offset and variable-to-variable addition                                               |
| `AveragePool`           | PyTorch `nn.AvgPool1d`, `nn.AdaptiveAvgPool1d`               | kernel_shape=1 pass-through; global mean for kernel_shape=[n_cols]; other configs raise `NotImplementedError` |
| `BatchNormalization`    | PyTorch `nn.BatchNorm1d`, sklearn-onnx pipelines             | Inference mode: `(x-mean)/sqrt(var+eps)*scale+bias`; constants frozen at export time                          |
| `ArgMax`                | multiclass classifiers                                       | Returns index of max value                                                                                    |
| `Cast` / `CastLike`     | type normalisation steps                                     |                                                                                                               |
| `Celu`                  | PyTorch `nn.CELU(alpha)`                                     | `max(0,x) + min(0, α*(exp(x/α)−1))`; default α=1                                                              |
| `Clip`                  | `relu6`, quantisation-aware models                           | Supports opset < 11 (attributes) and ≥ 11 (initializer inputs)                                                |
| `Concat`                | `ColumnTransformer`, `FeatureUnion`                          | axis=1 (column concat) only                                                                                   |
| `Div`                   | `StandardScaler` (scale)                                     |                                                                                                               |
| `Dropout`               | PyTorch Dropout layers                                       | Inference pass-through; training mask not used                                                                |
| `Elu`                   | PyTorch `nn.ELU(alpha)`                                      | `x ≥ 0 ? x : α*(exp(x)−1)`; default α=1                                                                       |
| `Erf`                   | GELU decomposition (opset ≤ 19, `approximate="none"`)        | A&S 7.1.28 polynomial approximation; max error ≈ 1.5×10⁻⁷                                                     |
| `Flatten`               | PyTorch MLP preprocessing                                    | axis=1 pass-through                                                                                           |
| `Gather`                | category indexing                                            |                                                                                                               |
| `Gelu`                  | PyTorch `nn.GELU()` (opset 20+)                              | Supports `approximate="tanh"` and `approximate="none"` modes                                                  |
| `Gemm`                  | `MLPRegressor`, `MLPClassifier`, `LinearRegression`          | Full `alpha·A@B + beta·C`, transA/transB                                                                      |
| `GlobalAveragePool`     | PyTorch `AdaptiveAvgPool`, pooling layers                    | Row-wise mean over all feature columns; 2D/3D not supported                                                   |
| `GlobalMaxPool`         | PyTorch `AdaptiveMaxPool`, pooling layers                    | Row-wise max over all feature columns via `ibis.greatest`                                                     |
| `GroupNormalization`    | PyTorch `nn.GroupNorm`                                       | Normalises within each group; num_groups=1 or num_groups=C                                                    |
| `HardSigmoid`           | PyTorch `nn.Hardsigmoid()`                                   | `max(0, min(1, α·x + β))`; α=0.2, β=0.5 from node attributes                                                  |
| `HardSwish`             | PyTorch `nn.Hardswish()`                                     | `x * max(0, min(1, (x+3)/6))`                                                                                 |
| `HardTanh`              | PyTorch `nn.Hardtanh()` exports                              | Equivalent to `Clip(x, -1, 1)`                                                                                |
| `Identity`              | no-op pass-through                                           |                                                                                                               |
| `InstanceNormalization` | PyTorch `nn.InstanceNorm1d`                                  | Per-row normalise across all channels; `(x-mean)/sqrt(var+eps)`                                               |
| `LayerNormalization`    | PyTorch `nn.LayerNorm`, keras3 `LayerNormalization`          | Per-row normalise using inline mean/var expressions (opset 17+)                                               |
| `LeakyRelu`             | PyTorch `nn.LeakyReLU(slope)`                                | `x ≥ 0 ? x : α·x`; α from node attribute, default α=0.01                                                      |
| `LogSigmoid`            | PyTorch `nn.LogSigmoid()` custom exports                     | `ln(sigmoid(x)) = -ln(1 + exp(-x))`                                                                           |
| `LogSoftmax`            | PyTorch log-probability output heads                         | Numerically stable log-sum-exp; axis=-1 / axis=1 only                                                         |
| `MatMul`                | general matrix multiply                                      |                                                                                                               |
| `MaxPool`               | PyTorch `nn.MaxPool1d`                                       | kernel_shape=1 pass-through; kernel_shape=[n_cols] global max; all other shapes raise `NotImplementedError`   |
| `Mish`                  | PyTorch `nn.Mish()`                                          | `x * tanh(softplus(x))`                                                                                       |
| `Mul`                   | element-wise multiply                                        |                                                                                                               |
| `Neg`                   | Unary negation; used in sigmoid `exp(-x)` decomposition      | Element-wise `-x`                                                                                             |
| `PRelu`                 | PyTorch `nn.PReLU()`, keras3 PReLU layer                     | Per-channel slope from `inputs[1]` initializer                                                                |
| `Pow`                   | GELU decomposition (`x³` term), polynomial activations       | Element-wise `a^b`                                                                                            |
| `ReduceMax`             | PyTorch `torch.max(..., dim=-1)` exports                     | Feature-axis reduction only (axis=-1 or axis=1)                                                               |
| `ReduceMean`            | PyTorch `torch.mean(..., dim=-1)` exports                    | Feature-axis reduction only (axis=-1 or axis=1)                                                               |
| `ReduceMin`             | PyTorch `torch.min(..., dim=-1)` exports                     | Feature-axis reduction only (axis=-1 or axis=1)                                                               |
| `ReduceSum`             | PyTorch `torch.sum(..., dim=-1)` exports                     | Feature-axis reduction only (axis=-1 or axis=1)                                                               |
| `Relu`                  | MLP hidden layers (`relu` activation)                        |                                                                                                               |
| `Reshape`               | see note                                                     | No-op when shape is compatible; raises for unsupported reshapes                                               |
| `Selu`                  | PyTorch `nn.SELU()`                                          | `γ·(x > 0 ? x : α·(exp(x)−1))`; γ≈1.0507, α≈1.6733; attributes can override                                   |
| `Sigmoid`               | `LogisticRegression`, binary MLP output                      |                                                                                                               |
| `Softmax`               | multiclass MLP output                                        | axis=-1 / axis=1                                                                                              |
| `Softplus`              | PyTorch `nn.Softplus()`                                      | `ln(1 + exp(x))`                                                                                              |
| `Softsign`              | PyTorch / keras Softsign exports                             | `x / (1 + \|x\|)`                                                                                             |
| `Sqrt`                  | Norm layer std dev: `sqrt(var + ε)`; utility op              | Element-wise square root                                                                                      |
| `Squeeze`               | ONNX tensor rank reduction                                   | Pass-through in the tabular single-row context                                                                |
| `Sub`                   | `StandardScaler` (mean)                                      |                                                                                                               |
| `Tanh`                  | MLP hidden layers (`tanh` activation)                        | Computed as `(exp(2x)-1)/(exp(2x)+1)` for SQL portability                                                     |
| `Transpose`             | PyTorch weight permutation                                   | 2-D pass-through                                                                                              |
| `Unsqueeze`             | ONNX tensor rank expansion                                   | Pass-through in the tabular single-row context                                                                |
| `Where`                 | conditional expressions                                      |                                                                                                               |

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
- **Spatial pooling** (`GlobalAveragePool`/`GlobalMaxPool` 2D/3D, `AveragePool`/`MaxPool` with stride > 1 or non-global 2D/3D kernels)

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

---

## PyTorch MLP workflow

The recommended way to use orbital with a PyTorch MLP is to export the model
to ONNX first, then translate:

```python
import torch
import torch.nn as nn
import orbital

# 1. Define and train your model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(64, 32),
    nn.ELU(alpha=1.0),
    nn.Linear(32, 1),
)
# ... train the model ...

# 2. Export to ONNX
model.eval()
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
)

# 3. Translate to SQL
feature_names = [f"x{i}" for i in range(10)]
sql_obj = orbital.translate("model.onnx", feature_names=feature_names)

# 4. Export SQL for any supported dialect
sql = orbital.export_sql("my_table", sql_obj, dialect="duckdb")
print(sql)
```

### Supported PyTorch activation ops

The following activations export cleanly to ONNX op-types that orbital
understands:

| PyTorch module                | ONNX op                 | orbital support                                       |
| ----------------------------- | ----------------------- | ----------------------------------------------------- |
| `nn.ReLU()`                   | `Relu`                  | ✅                                                    |
| `nn.Sigmoid()`                | `Sigmoid`               | ✅                                                    |
| `nn.Tanh()`                   | `Tanh`                  | ✅                                                    |
| `nn.ELU(alpha)`               | `Elu`                   | ✅                                                    |
| `nn.CELU(alpha)`              | `Celu`                  | ✅                                                    |
| `nn.SELU()`                   | `Selu`                  | ✅                                                    |
| `nn.LeakyReLU(slope)`         | `LeakyRelu`             | ✅                                                    |
| `nn.Hardsigmoid()`            | `HardSigmoid`           | ✅                                                    |
| `nn.Hardswish()`              | `HardSwish`             | ✅                                                    |
| `nn.Hardtanh()`               | `HardTanh`              | ✅ clips to [-1, 1]                                   |
| `nn.PReLU()`                  | `PRelu`                 | ✅ per-channel slope from initializer                 |
| `nn.Softplus()`               | `Softplus`              | ✅ `ln(1 + exp(x))`                                   |
| `nn.Mish()`                   | `Mish`                  | ✅ `x * tanh(softplus(x))`                            |
| `nn.LogSigmoid()`             | `LogSigmoid`            | ✅ `-ln(1 + exp(-x))`                                 |
| `nn.Softsign()`               | `Softsign`              | ✅ `x / (1 + \|x\|)`                                  |
| `nn.GELU(approximate="tanh")` | _(decomposed)_          | ✅ via `Tanh`/`Mul`/`Add`/`Pow` ops (opset ≤ 19)      |
| `nn.GELU(approximate="none")` | _(decomposed)_          | ✅ via `Erf`/`Mul`/`Add`/`Div` ops (opset ≤ 19)       |
| `nn.GELU()`                   | `Gelu`                  | ✅ native `Gelu` op (opset 20+); both modes supported |
| `nn.Dropout(p)`               | `Dropout`               | ✅ pass-through                                       |
| `nn.BatchNorm1d(n)`           | `BatchNormalization`    | ✅                                                    |
| `nn.InstanceNorm1d(n)`        | `InstanceNormalization` | ✅                                                    |
| `nn.LayerNorm(n)`             | `LayerNormalization`    | ✅ per-row symbolic normalisation (opset 17+)         |
| `nn.GroupNorm(g, n)`          | `GroupNormalization`    | ✅                                                    |
| `nn.AdaptiveAvgPool1d(1)`     | `GlobalAveragePool`     | ✅                                                    |
| `nn.AdaptiveMaxPool1d(1)`     | `GlobalMaxPool`         | ✅                                                    |
| `nn.MaxPool1d(k)`             | `MaxPool`               | ✅ (kernel=1 or global)                               |
| `nn.AvgPool1d(k)` / global    | `AveragePool`           | ✅ kernel=1 pass-through; global mean                 |
| `nn.LSTM` / `nn.GRU`          | `LSTM` / `GRU`          | ❌                                                    |
| `nn.MultiheadAttention`       | `MultiHeadAttention`    | ❌                                                    |

### Notes on GELU decomposition

PyTorch exports `nn.GELU()` as a sequence of primitive ONNX ops, not as a
single `Gelu` op (which only exists in opset 20+). Orbital handles both
decomposed forms:

- **`approximate="tanh"` (PyTorch default)**: uses `Tanh`/`Mul`/`Add`/`Pow`.  
  The approximation is $0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 x^3)))$.

- **`approximate="none"`**: uses `Erf`/`Mul`/`Add`/`Div`.  
  The `Erf` translator implements the Abramowitz & Stegun 7.1.26 polynomial approximation
  (max absolute error $\approx 1.5 \times 10^{-7}$).

For opset ≥ 20 models that export a native `Gelu` op, orbital now registers
a `GeluTranslator` that handles both `approximate="tanh"` and `approximate="none"` modes.
