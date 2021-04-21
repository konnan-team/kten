package eu.redbean.kten.opencl.tensor.store

import eu.redbean.kten.api.autograd.utils.reshape
import eu.redbean.kten.api.autograd.utils.unsqueeze

class OCLRawTensorView(
    var shape: List<Int>,
    val storeReference: OCLStoreView
) {

    fun unsqueeze(axis: Int): OCLRawTensorView {
        return OCLRawTensorView(shape.unsqueeze(axis), storeReference)
    }

    fun view(vararg newShape: Int): OCLRawTensorView {
        return OCLRawTensorView(shape.reshape(newShape.toList()), storeReference)
    }

}