package eu.redbean.kten.api.tensor.operations.nn

import eu.redbean.kten.api.tensor.store.AbstractRawTensor

data class PoolingResult<RAW_TYPE: AbstractRawTensor<*>>(
    val output: RAW_TYPE,
    val indices: RAW_TYPE?
)
