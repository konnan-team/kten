package eu.redbean.kten.api.tensor.operations.nn

import eu.redbean.kten.api.tensor.store.AbstractRawTensor

abstract class ConvolutionOperation<RAW_TYPE : AbstractRawTensor<*>> {

    abstract fun calculateOutput(
        input: RAW_TYPE,
        weight: RAW_TYPE,
        bias: RAW_TYPE?,
    ): RAW_TYPE

    abstract fun calculateGradInput(
        input: RAW_TYPE,
        gradOut: RAW_TYPE,
        weight: RAW_TYPE,
    ): RAW_TYPE

    abstract fun accumulateGradParams(
        input: RAW_TYPE,
        gradOut: RAW_TYPE,
        gradWeight: RAW_TYPE,
        gradBias: RAW_TYPE?,
        scale: Float = 1.0f
    )

    abstract fun cleanup()

}