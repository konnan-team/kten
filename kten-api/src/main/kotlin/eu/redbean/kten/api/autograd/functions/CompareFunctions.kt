package eu.redbean.kten.api.autograd.functions

import eu.redbean.kten.api.tensor.operations.TensorOperations
import eu.redbean.kten.api.tensor.store.AbstractRawTensor


abstract class CompareFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>,
    private val function: (AbstractRawTensor<Any>, AbstractRawTensor<Any>) -> AbstractRawTensor<Any>
): BiTensorFunction(ops, true) {

    override fun doForward(a: AbstractRawTensor<Any>, b: AbstractRawTensor<Any>) {
        output = function(a, b)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(
            if (inputs.first.requiresGrad) ops.createRaw(aShape) { 0f } else null,
            if (inputs.second.requiresGrad) ops.createRaw(bShape) { 0f } else null
        )
    }

}

class LessThan(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareFunction(ops, AbstractRawTensor<Any>::lt)

class LessThanEquals(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareFunction(ops, AbstractRawTensor<Any>::lte)

class GreaterThan(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareFunction(ops, AbstractRawTensor<Any>::gt)

class GreaterThanEquals(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareFunction(ops, AbstractRawTensor<Any>::gte)

class Equals(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareFunction(ops, AbstractRawTensor<Any>::eq)

class NotEquals(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareFunction(ops, AbstractRawTensor<Any>::neq)

abstract class CompareConstantFunction(
    ops: TensorOperations<AbstractRawTensor<Any>>,
    private val function: (AbstractRawTensor<Any>, Float) -> AbstractRawTensor<Any>
): TensorConstantFunction(ops) {

    override fun doForward(input: AbstractRawTensor<Any>) {
        output = function(input, constant)
    }

    override fun doBackward(gradient: AbstractRawTensor<Any>): List<AbstractRawTensor<Any>?> {
        return listOf(
            if (tensor.requiresGrad) ops.zerosLike(gradient) else null
        )
    }
}

class LessThanConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareConstantFunction(ops, AbstractRawTensor<Any>::lt)

class LessThanEqualsConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareConstantFunction(ops, AbstractRawTensor<Any>::lte)

class GreaterThanConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareConstantFunction(ops, AbstractRawTensor<Any>::gt)

class GreaterThanEqualsConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareConstantFunction(ops, AbstractRawTensor<Any>::gte)

class EqualsConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareConstantFunction(ops, AbstractRawTensor<Any>::eq)

class NotEqualsConstant(
    ops: TensorOperations<AbstractRawTensor<Any>>
): CompareConstantFunction(ops, AbstractRawTensor<Any>::neq)
