package eu.redbean.kten.opencl.tensor.store

import eu.redbean.kten.api.autograd.utils.*
import eu.redbean.kten.api.tensor.store.AbstractRawTensor
import eu.redbean.kten.api.tensor.store.AggregatingOperation
import eu.redbean.kten.api.tensor.store.RangeTensorIndexing
import eu.redbean.kten.api.tensor.store.SimpleTensorIndexing
import eu.redbean.kten.opencl.tensor.platform.OCLEnvironment
import eu.redbean.kten.opencl.tensor.platform.kernels.OCLKernelConstant
import eu.redbean.kten.opencl.tensor.platform.kernels.OCLKernelConstant.*
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.SOURCE
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.TARGET
import org.jocl.blast.CLBlast

class OCLRawTensor(
    shape: List<Int>,
    store: OCLMemoryObject,
    val environment: OCLEnvironment
) : AbstractRawTensor<OCLMemoryObject>(shape, store) {

    private var refCount = 1
    internal var mustSurviveGC = false

    init {
        environment.instanceCollector(this)
    }

    internal fun incrementRef() {
        if (storeReference.isReusable() || storeReference.isReleased()) {
            throw IllegalStateException("Store reference is already marked for reuse or release. (Tensor shape: ${shape})")
        }
        refCount++
    }

    internal fun referenced(): Boolean = refCount > 0

    internal fun release(): Boolean {
        if (!mustSurviveGC)
            refCount--

        if (refCount == 0) {
//            OCLPlatformInitializer.releaseExecutor.execute {
            environment.mayReuseOrRelease(storeReference)
//            }
            return true
        }
        return false
    }

    override fun broadcastTo(shape: List<Int>): AbstractRawTensor<OCLMemoryObject> {
        if (this.shape == shape)
            return this

        this.shape.inferExplicitBroadcastShape(shape)

        val thisShape = this.shape.toMutableList()
        for (i in 0 until (shape.size - thisShape.size)) {
            thisShape.add(0, 1)
        }

        val resStore = environment.memoryObject(shape.toStoreSize())
        environment.kernelStore.broadcastTo(storeReference, resStore, shape, thisShape)
        return OCLRawTensor(shape, resStore, environment)
    }

    override fun get(index: IntArray): AbstractRawTensor<OCLMemoryObject> {
        val (normIndex, resShape) = this.shape.normalizedIndexedShape(index)
        val range = SimpleTensorIndexing(this.shape, normIndex).storeRange
        return OCLRawTensor(resShape, storeReference.slice(range), environment)
    }

    override fun get(index: Array<out IntRange>): AbstractRawTensor<OCLMemoryObject> {
        val (normIndex, resShape) = this.shape.normalizedIndexedShape(index)
        val sparseRanges = RangeTensorIndexing(this.shape, normIndex).sparseStoreRanges
        return OCLRawTensor(resShape, storeReference.sparseSlices(sparseRanges), environment)
    }

    override fun set(index: IntArray, value: AbstractRawTensor<OCLMemoryObject>) {
        val range = this.shape.tensorIndexing(index).storeRange
        if (value.storeReference.size != (range.last - range.first + 1))
            throw IllegalArgumentException("Cannot set tensor with shape: ${value.shape} to tensor with shape: ${this.shape} at index: ${index.toList()}")

        storeReference.copyIntoAt(range.first, value.storeReference)
    }

    override fun set(index: Array<out IntRange>, value: AbstractRawTensor<OCLMemoryObject>) {
        val sparseRanges = this.shape.tensorIndexing(index).sparseStoreRanges
        val size = sparseRanges.map { it.last - it.first + 1 }.sum()

        if (size != value.storeReference.size)
            throw IllegalArgumentException("Cannot set tensor with shape ${value.shape} to tensor with shape: ${this.shape} at index: ${index.toList()}")

        storeReference.copyIntoSparseRanges(sparseRanges, value.storeReference)
    }

    override fun set(index: IntArray, value: Float) {
        val range = this.shape.tensorIndexing(index).storeRange
        environment.kernelStore.fill(storeReference, range, value)
    }

    override fun set(index: Array<out IntRange>, value: Float) {
        this.shape.tensorIndexing(index).sparseStoreRanges.forEach {
            environment.kernelStore.fill(storeReference, it, value)
        }
    }

    private fun elementwiseOpOnCommonShapedTensors(other: AbstractRawTensor<OCLMemoryObject>, op: OCLKernelConstant): OCLRawTensor {
        val commonShape = calculateCommonShapeWith(other)
        val t1 = this.broadcastTo(commonShape)
        val t2 = other.broadcastTo(commonShape)
        val res = environment.memoryObject(commonShape.toStoreSize())
        environment.kernelStore.elementwiseOpOnTensors(t1.storeReference, t2.storeReference, res, op)
        return OCLRawTensor(commonShape, res, environment)
    }

    private fun inplaceElementwiseOpOnTensors(other: AbstractRawTensor<OCLMemoryObject>, op: OCLKernelConstant) {
        if (this.shape != other.shape)
            throw IllegalArgumentException(
                "Shapes does not match for inplace operation on tensors with shapes: ${this.shape} and ${other.shape}."
            )

        environment.kernelStore.elementwiseAssignOpOnTensors(this.storeReference, other.storeReference, op)
    }

    private fun tensorConstantOp(constant: Float, op: OCLKernelConstant): OCLRawTensor {
        val res = environment.memoryObject(storeReference.size)
        environment.kernelStore.tensorConstOp(storeReference, res, constant, op)
        return OCLRawTensor(this.shape.toList(), res, environment)
    }

    private fun inplaceTensorConstantOp(constant: Float, op: OCLKernelConstant) {
        environment.kernelStore.tensorConstAssignOp(storeReference, constant, op)
    }

    override fun plus(other: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(other, PLUS)

    override fun plus(constant: Float) = tensorConstantOp(constant, PLUS)

    override fun plusAssign(other: AbstractRawTensor<OCLMemoryObject>) = inplaceElementwiseOpOnTensors(other, PLUS)

    override fun plusAssign(constant: Float) = inplaceTensorConstantOp(constant, PLUS)

    override fun minus(other: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(other, MINUS)

    override fun minus(constant: Float) = tensorConstantOp(constant, MINUS)

    override fun unaryMinus() = tensorConstantOp(-1f, TIMES)

    override fun times(other: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(other, TIMES)

    override fun timesAssign(other: AbstractRawTensor<OCLMemoryObject>) = inplaceElementwiseOpOnTensors(other, TIMES)

    override fun times(constant: Float) = tensorConstantOp(constant, TIMES)

    override fun timesAssign(constant: Float) = inplaceTensorConstantOp(constant, TIMES)

    override fun div(other: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(other, DIV)

    override fun div(constant: Float) = tensorConstantOp(constant, DIV)

    override fun divAssign(constant: Float) = inplaceTensorConstantOp(constant, DIV)

    override fun pow(other: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(other, POW)

    override fun pow(constant: Float) = tensorConstantOp(constant, POW)

    private fun tensorMapping(op: OCLKernelConstant): OCLRawTensor {
        val res = environment.memoryObject(storeReference.size)
        environment.kernelStore.tensorMappingOp(storeReference, res, op)
        return OCLRawTensor(this.shape.toList(), res, environment)
    }

    override fun reciprocal() = tensorMapping(RECIPROCAL)

    override fun log() = tensorMapping(LOG)

    private fun mapAggregatingOp(aggregatingOperation: AggregatingOperation): OCLKernelConstant {
        return when (aggregatingOperation) {
            AggregatingOperation.SUM -> SUM
            AggregatingOperation.MEAN -> MEAN
            AggregatingOperation.MAX -> MAX
            AggregatingOperation.MIN -> MIN
            AggregatingOperation.ARG_MAX -> ARG_MAX
            AggregatingOperation.ARG_MIN -> ARG_MIN
        }
    }

    override fun aggregateOverAxis(axis: Int, resShape: List<Int>, aggregatingOperation: AggregatingOperation): AbstractRawTensor<OCLMemoryObject> {
        val resSize = resShape.toStoreSize()
        val res = environment.memoryObject(resSize)
        environment.kernelStore.aggregateOverAxis(
            storeReference, res, shape, resShape, axis, mapAggregatingOp(aggregatingOperation)
        )
        return OCLRawTensor(resShape, res, environment)
    }

    override fun sum(): AbstractRawTensor<OCLMemoryObject> {
        val res = environment.memoryObject(1)
        environment.kernelStore.reductionOp(storeReference, res, SUM)
        return OCLRawTensor(listOf(1), res, environment)
    }

    override fun mean(): AbstractRawTensor<OCLMemoryObject> {
        val res = environment.memoryObject(1)
        environment.kernelStore.reductionOp(storeReference, res, MEAN)
        return OCLRawTensor(listOf(1), res, environment)
    }

    override fun inplaceScatter(axis: Int, index: AbstractRawTensor<OCLMemoryObject>, src: AbstractRawTensor<OCLMemoryObject>) {
        val normAxis = shape.normalizeAxis(axis)
        environment.kernelStore.inplaceScatter(
            this.storeReference, index.storeReference, src.storeReference,
            this.shape, index.shape, src.shape,
            normAxis
        )
    }

    override fun inplaceScatterAdd(axis: Int, index: AbstractRawTensor<OCLMemoryObject>, src: AbstractRawTensor<OCLMemoryObject>) {
        val normAxis = shape.normalizeAxis(axis)
        environment.kernelStore.inplaceScatterAdd(
            this.storeReference, index.storeReference, src.storeReference,
            this.shape, index.shape, src.shape,
            normAxis
        )
    }

    override fun inplaceScatter(axis: Int, index: AbstractRawTensor<OCLMemoryObject>, src: Float) {
        val normAxis = shape.normalizeAxis(axis)
        environment.kernelStore.inplaceScatterFill(
            this.storeReference, index.storeReference,
            this.shape, index.shape,
            normAxis, src
        )
    }

    override fun gather(axis: Int, index: AbstractRawTensor<OCLMemoryObject>): AbstractRawTensor<OCLMemoryObject> {
        val normAxis = shape.normalizeAxis(axis)
        val res = environment.memoryObject(index.storeReference.size)
        val resShape = index.shape.toList()
        environment.kernelStore.gather(
            this.storeReference, index.storeReference, res,
            this.shape, index.shape, resShape,
            normAxis
        )
        return OCLRawTensor(resShape, res, environment)
    }

    override fun getSingleValue(indexes: List<Int>): Float {
        return this.storeReference[this.shape.tensorIndexing(indexes).storeIndex]
    }

    override fun copy(shallow: Boolean): AbstractRawTensor<OCLMemoryObject> {
        if (shallow) {
            storeReference.incrementRef()
            return OCLRawTensor(shape.toList(), storeReference, environment)
        }
        return OCLRawTensor(shape.toList(), storeReference.copyOf(), environment)
    }

    override fun view(shape: List<Int>): AbstractRawTensor<OCLMemoryObject> {
        storeReference.incrementRef()
        return OCLRawTensor(this.shape.reshape(shape), storeReference, environment)
    }

    override fun lt(tensor: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(tensor, LT)

    override fun lte(tensor: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(tensor, LTE)

    override fun gt(tensor: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(tensor, GT)

    override fun gte(tensor: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(tensor, GTE)

    override fun eq(tensor: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(tensor, EQ)

    override fun neq(tensor: AbstractRawTensor<OCLMemoryObject>) = elementwiseOpOnCommonShapedTensors(tensor, NEQ)

    override fun lt(value: Float) = tensorConstantOp(value, LT)

    override fun lte(value: Float) = tensorConstantOp(value, LTE)

    override fun gt(value: Float) = tensorConstantOp(value, GT)

    override fun gte(value: Float) = tensorConstantOp(value, GTE)

    override fun eq(value: Float) = tensorConstantOp(value, EQ)

    override fun neq(value: Float) = tensorConstantOp(value, NEQ)

    override fun exp() = tensorMapping(EXP)

    override fun tanh() = tensorMapping(TANH)

    override fun sigmoid() = tensorMapping(SIGMOID)

    override fun sinh() = tensorMapping(SINH)

    override fun cosh() = tensorMapping(COSH)

    override fun abs() = tensorMapping(ABS)

    override fun sign() = tensorMapping(SIGN)

    override fun clamp(min: Float, max: Float): AbstractRawTensor<OCLMemoryObject> {
        val res = environment.memoryObject(storeReference.size)
        environment.kernelStore.clamp(storeReference, res, min, max)
        return OCLRawTensor(shape.toList(), res, environment)
    }

    override fun sqrt() = tensorMapping(SQRT)

    override fun sin() = tensorMapping(SIN)

    override fun cos() = tensorMapping(COS)

    override fun tan() = tensorMapping(TAN)

    override fun asin() = tensorMapping(ASIN)

    override fun acos() = tensorMapping(ACOS)

    override fun atan() = tensorMapping(ATAN)

    override fun floor() = tensorMapping(FLOOR)

    override fun ceil() = tensorMapping(CEIL)

    override fun round() = tensorMapping(ROUND)

    override fun trunc() = tensorMapping(TRUNC)

    override fun rsqrt() = tensorMapping(RSQRT)

    override fun transpose(axis1: Int, axis2: Int): AbstractRawTensor<OCLMemoryObject> {
        val (newShape, normAxis1, normAxis2) = shape.transposeNormalizeAxes(axis1, axis2)
        val res = environment.memoryObject(storeReference.size)
        environment.kernelStore.transpose(storeReference, res, shape, newShape, normAxis1, normAxis2)
        return OCLRawTensor(newShape, res, environment)
    }

    override fun permute(axes: List<Int>): AbstractRawTensor<OCLMemoryObject> {
        val newShape = shape.permute(axes)
        val res = environment.memoryObject(storeReference.size)
        environment.kernelStore.permute(storeReference, res, shape, newShape, axes)
        return OCLRawTensor(newShape, res, environment)
    }

    override fun narrow(axis: Int, start: Int, length: Int): AbstractRawTensor<OCLMemoryObject> {
        val normAxis = shape.normalizeAxis(axis) // TODO check start + length < shape[axis] in tensor impl.
        val ranges = shape.map { 0 until it }.toTypedArray()
        ranges[normAxis] = start until (start + length)
        return this[ranges]
    }

    override fun dot(vector: AbstractRawTensor<OCLMemoryObject>): AbstractRawTensor<OCLMemoryObject> {
        val resShape = shape.dotShape(vector.shape)
        val res = environment.memoryObject(1)
        CLBlast.CLBlastSdot(
            this.storeReference.size.toLong(),
            res.getMemoryObject(TARGET), 0,
            this.storeReference.getMemoryObject(SOURCE), 0, 1,
            vector.storeReference.getMemoryObject(SOURCE), 0, 1,
            environment.commandQueue, null
        )
        return OCLRawTensor(resShape, res, environment)
    }

    override fun indexSelect(axis: Int, index: AbstractRawTensor<OCLMemoryObject>): AbstractRawTensor<OCLMemoryObject> {
        val (normAxis, outputShape) = shape.indexSelectNormAxisShape(axis, index.shape)
        val output = environment.memoryObject(outputShape.toStoreSize())
        environment.kernelStore.indexSelect(storeReference, index.storeReference, output, shape, outputShape, index.shape[0], normAxis)
        return OCLRawTensor(outputShape, output, environment)
    }

    override fun indexAdd(axis: Int, index: AbstractRawTensor<OCLMemoryObject>, src: AbstractRawTensor<OCLMemoryObject>, alpha: Float) {
        val normAxis = shape.indexAddCheckShapesNormAxis(axis, index.shape, src.shape)
        environment.kernelStore.indexAdd(storeReference, index.storeReference, src.storeReference, shape, src.shape, index.shape[0], normAxis, alpha)
    }

    override fun containsNan(): Boolean {
        return storeReference.containsNan()
    }

    override fun inplaceResize(vararg shape: Int) {
        val newShape = shape.toList()
        val newStore = environment.memoryObject(newShape.toStoreSize())
        newStore.copyIntoAt(0, storeReference)
        environment.mayReuseOrRelease(storeReference)
        this.storeReference = newStore
        this.shape = newShape
    }

    override fun inplaceFill(value: Float) {
        storeReference.fill(value)
    }

    override fun maskedFill(mask: AbstractRawTensor<OCLMemoryObject>, value: Float): AbstractRawTensor<OCLMemoryObject> {
        val shapeCorrectedMask = if (mask.shape != shape) mask.broadcastTo(shape) else mask
        val res = environment.memoryObject(storeReference.size)
        environment.kernelStore.maskedFill(storeReference, shapeCorrectedMask.storeReference, res, shape, value)
        return OCLRawTensor(shape.toList(), res, environment)
    }

    fun view(vararg shape: Int): OCLRawTensorView {
        return OCLRawTensorView(this.shape.reshape(shape.toList()), OCLStoreView(0 until storeReference.size, storeReference))
    }

    fun asView(): OCLRawTensorView {
        return OCLRawTensorView(this.shape, OCLStoreView(0 until storeReference.size, storeReference))
    }

    fun getView(vararg index: Int): OCLRawTensorView {
        val (normIndex, resShape) = this.shape.normalizedIndexedShape(index)
        return OCLRawTensorView(resShape, OCLStoreView(SimpleTensorIndexing(shape, normIndex).storeRange, storeReference))
    }

}