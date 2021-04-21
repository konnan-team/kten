package eu.redbean.kten.api.autograd.functions.nn

import eu.redbean.kten.api.tensor.Tensor


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