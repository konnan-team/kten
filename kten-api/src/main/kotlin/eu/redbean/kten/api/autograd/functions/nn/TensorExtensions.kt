package eu.redbean.kten.api.autograd.functions.nn

import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.nn.PoolingOptions
import eu.redbean.kten.api.tensor.operations.nn.PoolingType
import eu.redbean.kten.api.tensor.operations.nn.UpsampleType


private fun checkedCall(function: ConvND, input: Tensor, weight: Tensor, bias: Tensor?): Tensor {
    if (listOf(input, weight, bias).any { it?.requiresGrad == true }) {
        return function(input, weight, bias)
    }

    return function(input, weight, bias).noGrad()
}

fun Tensor.conv1d(weight: Tensor,
                  bias: Tensor? = null,
                  stride: Int = 1,
                  padding: Int = 0,
                  dilation: Int = 1,
                  groups: Int = 1): Tensor {

    if (this.dimensions != 3)
        throw IllegalStateException("Conv1D cannot be applied to tensor with shape: $shape only 3D tensors allowed (batch x channels x W)")

    val function = ConvND(
        listOf(stride),
        listOf(padding),
        listOf(dilation),
        false,
        listOf(0),
        groups,
        this.platformOps()
    )

    return checkedCall(function, this, weight, bias)
}

fun Tensor.conv2d(weight: Tensor,
                  bias: Tensor? = null,
                  stride: Pair<Int, Int> = 1 to 1,
                  padding: Pair<Int, Int> = 0 to 0,
                  dilation: Pair<Int, Int> = 1 to 1,
                  groups: Int = 1): Tensor {

    if (this.dimensions != 4)
        throw IllegalStateException("Conv2d cannot be applied to tensor with shape: $shape only 4D tensors allowed (batch x channels x H x W)")

    val function = ConvND(
        stride.toList(),
        padding.toList(),
        dilation.toList(),
        false,
        listOf(0, 0),
        groups,
        this.platformOps()
    )

    return checkedCall(function, this, weight, bias)
}

fun Tensor.conv3d(weight: Tensor,
                  bias: Tensor? = null,
                  stride: Triple<Int, Int, Int> = Triple(1, 1, 1),
                  padding: Triple<Int, Int, Int> = Triple(0, 0, 0),
                  dilation: Triple<Int, Int, Int> = Triple(1, 1, 1),
                  groups: Int = 1): Tensor {

    if (this.dimensions != 5)
        throw IllegalStateException("Conv3d cannot be applied to tensor with shape: $shape only 5D tensors allowed (batch x channels x D x H x W)")

    val function = ConvND(
        stride.toList(),
        padding.toList(),
        dilation.toList(),
        false,
        listOf(0, 0, 0),
        groups,
        this.platformOps()
    )

    return checkedCall(function, this, weight, bias)
}

fun Tensor.conv1dTranspose(weight: Tensor,
                           bias: Tensor? = null,
                           stride: Int = 1,
                           padding: Int = 0,
                           outputPadding: Int = 0,
                           dilation: Int = 1,
                           groups: Int = 1): Tensor {

    if (this.dimensions != 3)
        throw IllegalStateException("Transposed Conv1D cannot be applied to tensor with shape: $shape only 3D tensors allowed (batch x channels x W)")

    val function = ConvND(
        listOf(stride),
        listOf(padding),
        listOf(dilation),
        true,
        listOf(outputPadding),
        groups,
        this.platformOps()
    )

    return checkedCall(function, this, weight, bias)
}

fun Tensor.conv2dTranspose(weight: Tensor,
                           bias: Tensor? = null,
                           stride: Pair<Int, Int> = 1 to 1,
                           padding: Pair<Int, Int> = 0 to 0,
                           outputPadding: Pair<Int, Int> = 0 to 0,
                           dilation: Pair<Int, Int> = 1 to 1,
                           groups: Int = 1): Tensor {

    if (this.dimensions != 4)
        throw IllegalStateException("Transposed Conv2d cannot be applied to tensor with shape: $shape only 4D tensors allowed (batch x channels x H x W)")

    val function = ConvND(
        stride.toList(),
        padding.toList(),
        dilation.toList(),
        true,
        outputPadding.toList(),
        groups,
        this.platformOps()
    )

    return checkedCall(function, this, weight, bias)
}

fun Tensor.conv3dTranspose(weight: Tensor,
                           bias: Tensor? = null,
                           stride: Triple<Int, Int, Int> = Triple(1, 1, 1),
                           padding: Triple<Int, Int, Int> = Triple(0, 0, 0),
                           outputPadding: Triple<Int, Int, Int> = Triple(0, 0, 0),
                           dilation: Triple<Int, Int, Int> = Triple(1, 1, 1),
                           groups: Int = 1): Tensor {

    if (this.dimensions != 5)
        throw IllegalStateException("Transposed Conv3d cannot be applied to tensor with shape: $shape only 5D tensors allowed (batch x channels x D x H x W)")

    val function = ConvND(
        stride.toList(),
        padding.toList(),
        dilation.toList(),
        true,
        outputPadding.toList(),
        groups,
        this.platformOps()
    )

    return checkedCall(function, this, weight, bias)
}

fun Tensor.maxPooling1d(kernel: Int, stride: Int = kernel, padding: Int = 0, dilation: Int = 1, useCeil: Boolean = false): Tensor {
    if (this.dimensions != 3) {
        throw IllegalStateException("1D pooling cannot be applied to tensor with shape: $shape " +
                "only tensors with 3 dimensions allowed (batch x channels x elements)")
    }

    val function = Pooling2DOperation(
        this.platformOps(),
        listOf(kernel, 1),
        listOf(stride, 1),
        listOf(padding, 0),
        listOf(dilation, 1),
        PoolingOptions(PoolingType.MAX, useCeil)
    )

    val input = this.unsqueeze(3)
    return (if (requiresGrad) function(input) else function(input).noGrad()).squeeze(3)
}

fun Tensor.maxPooling2d(kernel: Pair<Int, Int>,
                        stride: Pair<Int, Int> = kernel,
                        padding: Pair<Int, Int> = 0 to 0,
                        dilation: Pair<Int, Int> = 1 to 1,
                        useCeil: Boolean = false): Tensor {
    val function = Pooling2DOperation(
        this.platformOps(),
        kernel.toList(),
        stride.toList(),
        padding.toList(),
        dilation.toList(),
        PoolingOptions(PoolingType.MAX, useCeil)
    )

    return if (requiresGrad) function(this) else function(this).noGrad()
}

fun Tensor.avgPooling1d(kernel: Int,
                        stride: Int = kernel,
                        padding: Int = 0,
                        useCeil: Boolean = false,
                        includePadding: Boolean = false): Tensor {
    if (this.dimensions != 3) {
        throw IllegalStateException("1D pooling cannot be applied to tensor with shape: $shape " +
                "only tensors with 3 dimensions allowed (batch x channels x elements)")
    }

    val function = Pooling2DOperation(
        this.platformOps(),
        listOf(kernel, 1),
        listOf(stride, 1),
        listOf(padding, 0),
        listOf(1, 1),
        PoolingOptions(PoolingType.AVG, useCeil, includePadding)
    )

    val input = this.unsqueeze(3)
    return (if (requiresGrad) function(input) else function(input).noGrad()).squeeze(3)
}

fun Tensor.avgPooling2d(kernel: Pair<Int, Int>,
                        stride: Pair<Int, Int> = kernel,
                        padding: Pair<Int, Int> = 0 to 0,
                        useCeil: Boolean = false,
                        includePadding: Boolean = false): Tensor {
    val function = Pooling2DOperation(
        this.platformOps(),
        kernel.toList(),
        stride.toList(),
        padding.toList(),
        listOf(1, 1),
        PoolingOptions(PoolingType.AVG, useCeil, includePadding)
    )

    return if (requiresGrad) function(this) else function(this).noGrad()
}

fun Tensor.batchNorm(axis: Int,
                     runningMean: Tensor?,
                     runningVar: Tensor?,
                     training: Boolean = false,
                     momentum: Float = 0.99f,
                     epsilon: Float = 1e-3f,
                     gamma: Tensor? = null,
                     beta: Tensor? = null): Tensor {
    val function = BatchNormalizationFunction(axis, momentum, epsilon, training, this.platformOps())
    val res = function(this, runningMean, runningVar, gamma, beta)
    return if (requiresGrad) res else res.noGrad()
}

fun Tensor.upsample2d(upsampleType: UpsampleType, scale: Int): Tensor {
    val function = Upsample2DFunction(upsampleType, scale, this.platformOps())
    return if (requiresGrad) function(this) else function(this).noGrad()
}

fun Tensor.softmaxCrossEntropy(targets: Tensor): Tensor {
    val function = SoftmaxCrossEntropy(this.platformOps())
    return if (requiresGrad) function(this, targets) else function(this, targets).noGrad()
}
