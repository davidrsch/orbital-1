"""Registry mapping ONNX operation names to their Orbital translator classes."""

from .steps.abs import AbsTranslator
from .steps.add import AddTranslator
from .steps.argmax import ArgMaxTranslator
from .steps.arrayfeatureextractor import ArrayFeatureExtractorTranslator
from .steps.avgpool import AveragePoolTranslator
from .steps.batchnorm import BatchNormalizationTranslator
from .steps.cast import CastTranslator
from .steps.castlike import CastLikeTranslator
from .steps.celu import CeluTranslator
from .steps.conv import ConvTranslator
from .steps.convtranspose import ConvTransposeTranslator
from .steps.gru import GRUTranslator
from .steps.lstm import LSTMTranslator
from .steps.clip import ClipTranslator
from .steps.concat import ConcatTranslator
from .steps.featurevectorizer import FeatureVectorizerTranslator
from .steps.div import DivTranslator
from .steps.dropout import DropoutTranslator
from .steps.einsum import EinsumTranslator
from .steps.elu import EluTranslator
from .steps.ceil import CeilTranslator
from .steps.erf import ErfTranslator
from .steps.exp import ExpTranslator
from .steps.flatten import FlattenTranslator
from .steps.floor import FloorTranslator
from .steps.log import LogTranslator
from .steps.gather import GatherTranslator
from .steps.gelu import GeluTranslator
from .steps.gemm import GemmTranslator
from .steps.globalavgpool import GlobalAveragePoolTranslator
from .steps.globalmaxpool import GlobalMaxPoolTranslator
from .steps.groupnorm import GroupNormalizationTranslator
from .steps.hardsigmoid import HardSigmoidTranslator
from .steps.hardswish import HardSwishTranslator
from .steps.hardtanh import HardTanhTranslator
from .steps.identity import IdentityTranslator
from .steps.imputer import ImputerTranslator
from .steps.isinf import IsInfTranslator
from .steps.isnan import IsNaNTranslator
from .steps.instancenorm import InstanceNormalizationTranslator
from .steps.labelencoder import LabelEncoderTranslator
from .steps.layernorm import LayerNormalizationTranslator
from .steps.leakyrelu import LeakyReluTranslator
from .steps.localresponsenormalization import LRNTranslator
from .steps.logsigmoid import LogSigmoidTranslator
from .steps.logsoftmax import LogSoftmaxTranslator
from .steps.lpnormalization import LpNormalizationTranslator
from .steps.linearclass import LinearClassifierTranslator
from .steps.linearreg import LinearRegressorTranslator
from .steps.matmul import MatMulTranslator
from .steps.maxpool import MaxPoolTranslator
from .steps.mish import MishTranslator
from .steps.mod import ModTranslator
from .steps.meanvariancenorm import MeanVarianceNormalizationTranslator
from .steps.attention import AttentionTranslator
from .steps.multiheadattention import MultiHeadAttentionTranslator
from .steps.skip_layer_norm import SkipLayerNormalizationTranslator
from .steps.mul import MulTranslator
from .steps.neg import NegTranslator
from .steps.onehotencoder import OneHotEncoderTranslator
from .steps.pad import PadTranslator
from .steps.pow import PowTranslator
from .steps.prelu import PReluTranslator
from .steps.reducemax import ReduceMaxTranslator
from .steps.reducemean import ReduceMeanTranslator
from .steps.reducemin import ReduceMinTranslator
from .steps.reducesum import ReduceSumTranslator
from .steps.reducel1 import ReduceL1Translator
from .steps.reducel2 import ReduceL2Translator
from .steps.reducelogsum import ReduceLogSumTranslator
from .steps.reducelogsumexp import ReduceLogSumExpTranslator
from .steps.reducesumsquare import ReduceSumSquareTranslator
from .steps.variadicsum import VariadicSumTranslator
from .steps.variadicmax import VariadicMaxTranslator
from .steps.variadicmin import VariadicMinTranslator
from .steps.variadicmean import VariadicMeanTranslator
from .steps.reduceprod import ReduceProdTranslator
from .steps.relu import ReluTranslator
from .steps.reshape import ReshapeTranslator
from .steps.round_ import RoundTranslator
from .steps.rmsnorm import RMSNormalizationTranslator
from .steps.rnn import RNNTranslator
from .steps.scaler import ScalerTranslator
from .steps.sign import SignTranslator
from .steps.shrink import ShrinkTranslator
from .steps.scatterelements import ScatterElementsTranslator
from .steps.shape import ShapeTranslator
from .steps.split import SplitTranslator
from .steps.selu import SeluTranslator
from .steps.sigmoid import SigmoidTranslator
from .steps.softmax import SoftmaxTranslator
from .steps.softplus import SoftplusTranslator
from .steps.softsign import SoftsignTranslator
from .steps.sqrt import SqrtTranslator
from .steps.squeeze import SqueezeTranslator
from .steps.unsqueeze import UnsqueezeTranslator
from .steps.sub import SubTranslator
from .steps.swish import SwishTranslator
from .steps.tanh import TanhTranslator
from .steps.thresholdedrelu import ThresholdedReluTranslator
from .steps.slice import SliceTranslator
from .steps.tile import TileTranslator
from .steps.trilu import TriluTranslator
from .steps.transpose import TransposeTranslator
from .steps.trees import (
    TreeEnsembleClassifierTranslator,
    TreeEnsembleRegressorTranslator,
)
from .steps.where import WhereTranslator
from .steps.zipmap import ZipMapTranslator
from .translator import Translator

# This is a mapping of ONNX operations to their respective translators.
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
    "Einsum": EinsumTranslator,
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
    # Variadic elementwise ops (multi-input; key matches ONNX op name)
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
    "Trilu": TriluTranslator,
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
