package eu.redbean.kten.api.tensor.operations.nn

import eu.redbean.kten.api.tensor.store.AbstractRawTensor

interface PoolingOperation<RAW_TYPE: AbstractRawTensor<*>> {

    fun updateOutput(input: RAW_TYPE): PoolingResult<RAW_TYPE>

    fun calculateGradInput(input: RAW_TYPE, gradOut: RAW_TYPE, indices: RAW_TYPE?): RAW_TYPE

    companion object {
        fun calculateSizeOnFloats(
            input: Float, dilation: Float, kernel: Float, padding: Float, stride: Float
        ) = (input - (dilation * (kernel - 1f) + 1f) + 2f * padding) / stride + 1f
    }
}