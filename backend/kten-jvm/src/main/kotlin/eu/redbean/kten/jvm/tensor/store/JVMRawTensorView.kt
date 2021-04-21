package eu.redbean.kten.jvm.tensor.store

import eu.redbean.kten.api.autograd.utils.reshape
import eu.redbean.kten.api.autograd.utils.unsqueeze

class JVMRawTensorView(
    var shape: List<Int>,
    val storeReference: StoreView,
) {

    fun unsqueeze(axis: Int): JVMRawTensorView {
        return JVMRawTensorView(shape.unsqueeze(axis), storeReference)
    }

    operator fun timesAssign(value: Float) {
        for (i in 0 until storeReference.size)
            storeReference[i] *= value
    }

    fun view(vararg newShape: Int): JVMRawTensorView {
        return JVMRawTensorView(shape.reshape(newShape.toList()), storeReference)
    }

}