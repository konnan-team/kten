package eu.redbean.kten.api.tensor.operations

import eu.redbean.kten.api.tensor.Tensor
import java.io.Closeable

interface BasicTensorOperations {

    val platformKey: String

    fun create(shape: List<Int>, requiresGrad: Boolean = false, init: (Int) -> Float = { 0.0f }): Tensor

    fun createFillConst(shape: List<Int>, requiresGrad: Boolean = false, constant: Float): Tensor = create(shape, requiresGrad) { constant }

    fun createRandom(shape: List<Int>, requiresGrad: Boolean = false): Tensor

    fun createRandom(vararg shape: Int, requiresGrad: Boolean = false): Tensor {
        return createRandom(shape.toList(), requiresGrad)
    }

    fun createBernoulli(shape: List<Int>, rate: Float, requiresGrad: Boolean = false): Tensor

    fun garbageCollector(): TensorOperationsGarbageCollector

}