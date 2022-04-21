package eu.redbean.kten.api.tensor.operations.nn

import eu.redbean.kten.api.tensor.store.AbstractRawTensor

interface Upsample2DOperation<RAW_TYPE: AbstractRawTensor<*>> {

    val scale: Int

    fun checkDimensions(inputShape: List<Int>) {
        if (inputShape.size !in 3..4) {
            throw IllegalArgumentException("2D Upsampling can only be applied to tensors with 3 or 4 dimensions, but got tensor with shape: $inputShape")
        }
    }

    fun calculateOutputShape(inputShape: List<Int>) =
        inputShape.dropLast(2) + (inputShape[inputShape.size - 2] * scale) + (inputShape[inputShape.size - 1] * scale)

    fun checkGradOutShape(gradOutShape: List<Int>, inputShape: List<Int>) {
        if (inputShape.size != gradOutShape.size) {
            throw IllegalArgumentException("2D Upsample got gradient with different dimensions (${gradOutShape.size}) " +
                    "as the input tensor dimensions (${inputShape.size})")
        }

        if (gradOutShape[gradOutShape.size - 2] % scale != 0 || gradOutShape[gradOutShape.size - 1] % scale != 0) {
            throw IllegalArgumentException("2D Upsample output gradients shape must be divisible by scale: $scale in the last two dimensions," +
                    "but got gradient tensor with shape: $gradOutShape")
        }

        if (gradOutShape[gradOutShape.size - 2] / scale != inputShape[inputShape.size - 2]
            || gradOutShape[gradOutShape.size - 1] / scale != inputShape[inputShape.size - 1]
            || gradOutShape.dropLast(2) != inputShape.dropLast(2)) {
            throw IllegalArgumentException("2D Upsample output gradients shape: $gradOutShape does not align with input tensor shape: $inputShape " +
                    "with scale: $scale (output gradients shape's last two dimensions divided by the scale must equal to the input tensors shape " +
                    "in the last two dimensions, and in all other dimensions the shape must be the same)")
        }
    }

    fun upsample(input: RAW_TYPE): RAW_TYPE

    fun calculateGrad(gradOut: RAW_TYPE, inputShape: List<Int>): RAW_TYPE

}