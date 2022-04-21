package eu.redbean.kten.jvm.tensor.operations.nn

import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.operations.nn.Upsample2DOperation
import eu.redbean.kten.jvm.tensor.operations.AbstractJVMTensorOperations
import eu.redbean.kten.jvm.tensor.store.JVMRawTensor
import java.util.stream.IntStream

class JVMUpsample2DNearest(
    override val scale: Int,
    val ops: AbstractJVMTensorOperations
) : Upsample2DOperation<JVMRawTensor> {

    override fun upsample(input: JVMRawTensor): JVMRawTensor {
        checkDimensions(input.shape)
        val inputTensor = if (input.dimensions == 3) input.view(listOf(1) + input.shape) else input
        val outputShape = calculateOutputShape(inputTensor.shape)
        val outputTensor = ops.createRaw(outputShape)

        IntStream.range(0, outputShape.toStoreSize()).parallel().forEach {
            val outStride0 = outputShape[3] * outputShape[2] * outputShape[1]
            val inStride0 = (outputShape[3] / scale) * (outputShape[2] / scale) * outputShape[1]
            val outStride1 = outputShape[3] * outputShape[2]
            val inStride1 = (outputShape[3] / scale) * (outputShape[2] / scale)
            val outStride2 = outputShape[3]
            val inStride2 = outputShape[3] / scale

            val iin0 = it / outStride0
            val iin1 = ((it / outStride1) % outputShape[1])
            val iin2 = ((it / outStride2) % outputShape[2]) / scale
            val iin3 = it % outputShape[3] / scale

            val isrc = iin0 * inStride0 + iin1 * inStride1 + iin2 * inStride2 + iin3

            outputTensor.storeReference[it] = inputTensor.storeReference[isrc]
        }

        if (input.dimensions == 3)
            outputTensor.inplaceReshape(outputTensor.shape.drop(1))

        return outputTensor
    }

    override fun calculateGrad(gradOut: JVMRawTensor, inputShape: List<Int>): JVMRawTensor {
        checkDimensions(gradOut.shape)
        checkGradOutShape(gradOut.shape, inputShape)
        val gradOutTensor = if (gradOut.dimensions == 3) gradOut.view(listOf(1) + gradOut.shape) else gradOut
        val gradInShape = if (inputShape.size == 3) listOf(1) + inputShape else inputShape
        val gradInTensor = ops.createRawFill(gradInShape, 0f)

        IntStream.range(0, gradInShape.toStoreSize()).parallel().forEach {
            val inStride0 = gradInShape[3] * gradInShape[2] * gradInShape[1]
            val outStride0 = gradInShape[3] * scale * gradInShape[2] * scale * gradInShape[1]
            val inStride1 = gradInShape[3] * gradInShape[2]
            val outStride1 = gradInShape[3] * scale * gradInShape[2] * scale
            val inStride2 = gradInShape[3]
            val outStride2 = gradInShape[3] * scale

            val iin0 = it / inStride0
            val iin1 = ((it / inStride1) % gradInShape[1])
            val iin2 = ((it / inStride2) % gradInShape[2])
            val iin3 = it % gradInShape[3]

            for (y in 0 until scale) {
                for (x in 0 until scale) {
                    val ioutx = iin2 * scale + x
                    val iouty = iin3 * scale + y
                    val isrc = iin0 * outStride0 + iin1 * outStride1 + ioutx * outStride2 + iouty
                    gradInTensor.storeReference[it] += gradOutTensor.storeReference[isrc]
                }
            }
        }

        if (inputShape.size == 3)
            gradInTensor.inplaceReshape(inputShape)

        return gradInTensor
    }

}