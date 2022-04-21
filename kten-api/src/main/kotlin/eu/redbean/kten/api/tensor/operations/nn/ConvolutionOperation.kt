package eu.redbean.kten.api.tensor.operations.nn

import eu.redbean.kten.api.tensor.store.AbstractRawTensor

interface ConvolutionOperation<RAW_TYPE : AbstractRawTensor<*>> {

    fun calculateOutput(
        input: RAW_TYPE,
        weight: RAW_TYPE,
        bias: RAW_TYPE?,
    ): RAW_TYPE

    fun calculateGradInput(
        input: RAW_TYPE,
        gradOut: RAW_TYPE,
        weight: RAW_TYPE,
    ): RAW_TYPE

    fun accumulateGradParams(
        input: RAW_TYPE,
        gradOut: RAW_TYPE,
        gradWeight: RAW_TYPE,
        gradBias: RAW_TYPE?,
        scale: Float = 1.0f
    )

    fun cleanup()

}