package eu.redbean.kten.api.autograd.functions.nn

import eu.redbean.kten.api.autograd.functions.UnaryTensorFunction
import eu.redbean.kten.api.autograd.tensor.AGTensor
import eu.redbean.kten.api.tensor.Tensor
import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.operations.nn.PoolingOperation
import eu.redbean.kten.api.tensor.operations.nn.PoolingOptions
import eu.redbean.kten.api.tensor.store.AbstractRawTensor

class Pooling2DOperation(
    ops: TensorOperations<AbstractRawTensor<Any>>,
    val kernelSize: List<Int>,
    val stride: List<Int>,
    val padding: List<Int>,
    val dilation: List<Int>,
    val poolingOptions: PoolingOptions
) : UnaryTensorFunction(ops) {

    private lateinit var poolingOp: PoolingOperation<AbstractRawTensor<Any>>

    override operator fun invoke(tensor: Tensor): Pooling2DOperation {
        modifiesShape = true
        super.invoke(if (tensor is AGTensor) tensor.inplaceSafe() else tensor)

        if (tensor.dimensions !in listOf(3, 4)) {
            throw IllegalArgumentException(
                "2D pooling requires input tensor with 3 or 4 dimensions, but got tensor with shape: ${tensor.shape}"
            )
        }

        if (tensor.dimensions == 3) {
            cachedShape = listOf(tensor.shape[0]) + calcOutHeightWidth(tensor.shape)
        } else {
            cachedShape = listOf(tensor.shape[0], tensor.shape[1]) + calcOutHeightWidth(tensor.shape)
        }

        poolingOp = ops.spatialPooling(kernelSize, padding, stride, dilation, poolingOptions)

        return this
    }

    private fun calcOutHeightWidth(inputShape: List<Int>): List<Int> {
        return if (poolingOptions.useCeil) {
            listOf(
                kotlin.math.ceil(
                    PoolingOperation.calculateSizeOnFloats(
                        inputShape[inputShape.size - 2].toFloat(), dilation[0].toFloat(), kernelSize[0].toFloat(), padding[0].toFloat(), stride[0].toFloat()
                    )
                ).toInt(),
                kotlin.math.ceil(
                    PoolingOperation.calculateSizeOnFloats(
                        inputShape[inputShape.size - 1].toFloat(), dilation[1].toFloat(), kernelSize[1].toFloat(), padding[1].toFloat(), stride[1].toFloat()
                    )
                ).toInt()
            )
        } else {
            listOf(
                kotlin.math.floor(
                    PoolingOperation.calculateSizeOnFloats(
                        inputShape[inputShape.size - 2].toFloat(), dilation[0].toFloat(), kernelSize[0].toFloat(), padding[0].toFloat(), stride[0].toFloat()
                    )
                ).toInt(),
                kotlin.math.floor(
                    PoolingOperation.calculateSizeOnFloats(
                        inputShape[inputShape.size - 1].toFloat(), dilation[1].toFloat(), kernelSize[1].toFloat(), padding[1].toFloat(), stride[1].toFloat()
                    )
                ).toInt()
            )
        }
    }

    override fun doForward(input: AbstractRawTensor<Any>) {
        val (outTensor, indices) = poolingOp.updateOutput(input)
        if (indices != null) {
            saveForBackward(input, indices)
        } else {
            saveForBackward(input)
        }
        this.output = outTensor
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        val input = valuesSaved[0]
        val indices = if (valuesSaved.size == 2) valuesSaved[1] else null
        return listOf(poolingOp.calculateGradInput(input, gradient, indices))
    }

}
