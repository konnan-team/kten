package eu.redbean.kten.api.tensor.operations.nn

import eu.redbean.kten.api.tensor.store.AbstractRawTensor

interface BatchNormOperation<RAW_TYPE : AbstractRawTensor<*>> {

    fun calculateOutput(
        input: RAW_TYPE,
        runningMean: RAW_TYPE?,
        runningVar: RAW_TYPE?,
        gamma: RAW_TYPE?,
        beta: RAW_TYPE?
    ): BatchNormOutputs<RAW_TYPE>

    fun calculateGrads(
        input: RAW_TYPE,
        runningMean: RAW_TYPE?,
        runningVar: RAW_TYPE?,
        currentMean: RAW_TYPE,
        currentStd: RAW_TYPE,
        gamma: RAW_TYPE?,
        gradOut: RAW_TYPE,
        inRequiresGrad: Boolean,
        gammaRequiresGrad: Boolean,
        betaRequiresGrad: Boolean
    ): BatchNormGrads<RAW_TYPE>


    data class BatchNormOutputs<RAW_TYPE : AbstractRawTensor<*>>(
        val output: RAW_TYPE,
        val currentMean: RAW_TYPE,
        val currentStd: RAW_TYPE
    )


    data class BatchNormGrads<RAW_TYPE : AbstractRawTensor<*>>(
        val gradIn: RAW_TYPE?,
        val gradGamma: RAW_TYPE?,
        val gradBeta: RAW_TYPE?
    )

}