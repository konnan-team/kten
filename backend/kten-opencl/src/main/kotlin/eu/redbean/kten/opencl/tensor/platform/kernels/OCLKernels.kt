package eu.redbean.kten.opencl.tensor.platform.kernels

import eu.redbean.kten.api.autograd.utils.toStoreSize
import eu.redbean.kten.api.tensor.platform.PlatformProvider
import eu.redbean.kten.opencl.tensor.platform.OCLPlatformInitializer
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject
import eu.redbean.kten.opencl.tensor.store.OCLMemoryObject.MemoryAccessOption.*
import eu.redbean.kten.opencl.tensor.store.OCLStoreView
import org.jocl.*
import org.jocl.blast.CLBlast
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

data class OCLKernelDescriptor(
    val kernel: cl_kernel,
    val context: cl_context,
    val commandQueue: cl_command_queue
)

annotation class KernelName(val functionName: String, val kotlinName: String)

abstract class AbstractOCLKernel(
    val descriptor: OCLKernelDescriptor
) {

    private fun createHostCopyConstMemObject(array: IntArray, isResult: Boolean = false): cl_mem {
        val pointer = Pointer.to(array)
        val memSize = Sizeof.cl_int * array.size.toLong()
        return CL.clCreateBuffer(
            descriptor.context,
            if (isResult) CL.CL_MEM_READ_WRITE or CL.CL_MEM_COPY_HOST_PTR else CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR,
            memSize,
            pointer,
            null
        )
    }

    private fun readMemObjTo(array: IntArray, memObject: cl_mem, commandQueue: cl_command_queue) {
        CL.clEnqueueReadBuffer(
            commandQueue,
            memObject,
            CL.CL_TRUE,
            0,
            array.size.toLong() * Sizeof.cl_int,
            Pointer.to(array),
            0,
            null,
            null
        )
    }

    fun runSingleDimKernel(workSize: Long, vararg kernelArguments: Any) {
        with(descriptor) {
            val memObjectsToRelease = mutableListOf<cl_mem>()
            var resultArray: IntArray? = null
            var resultIndex = -1
            var resultMemObject: cl_mem? = null

            for (i in kernelArguments.indices) {
                val arg = kernelArguments[i]
                if (arg is cl_mem) {
                    CL.clSetKernelArg(kernel, i, Sizeof.cl_mem.toLong(), Pointer.to(arg))
                } else if (arg is Float) {
                    CL.clSetKernelArg(kernel, i, Sizeof.cl_float.toLong(), Pointer.to(FloatArray(1) { arg }))
                } else if (arg is IntArray) {
                    val memObject = createHostCopyConstMemObject(arg)
                    memObjectsToRelease.add(memObject)
                    CL.clSetKernelArg(kernel, i, Sizeof.cl_mem.toLong(), Pointer.to(memObject))
                } else if (arg is Int) {
                    CL.clSetKernelArg(kernel, i, Sizeof.cl_int.toLong(), Pointer.to(IntArray(1) { arg }))
                } else if (arg is Long) {
                    CL.clSetKernelArg(kernel, i, Sizeof.cl_long.toLong(), Pointer.to(LongArray(1) { arg }))
                } else if (arg is AtomicBoolean) { //TODO maybe this could be nicer
                    resultArray = IntArray(1) { if (arg.get()) 1 else 0 }
                    resultIndex = i
                    resultMemObject = createHostCopyConstMemObject(resultArray, true)
                    memObjectsToRelease.add(resultMemObject)
                    CL.clSetKernelArg(kernel, i, Sizeof.cl_mem.toLong(), Pointer.to(resultMemObject))
                } else if (arg is AtomicInteger) {
                    resultArray = IntArray(1) { arg.get() }
                    resultIndex = i
                    resultMemObject = createHostCopyConstMemObject(resultArray, true)
                    memObjectsToRelease.add(resultMemObject)
                    CL.clSetKernelArg(kernel, i, Sizeof.cl_mem.toLong(), Pointer.to(resultMemObject))
                }
            }

            CL.clEnqueueNDRangeKernel(
                commandQueue,
                kernel,
                1, null, LongArray(1) { workSize }, null,
                0, null,
                null
            )

            if (resultIndex != -1 && resultArray != null && resultMemObject != null) {
                readMemObjTo(resultArray, resultMemObject, commandQueue)
                val resArg = kernelArguments[resultIndex]
                if (resArg is AtomicBoolean) {
                    resArg.set(resultArray[0] == 1)
                } else if (resArg is AtomicInteger) {
                    resArg.set(resultArray[0])
                }
            }

            memObjectsToRelease.forEach {
                OCLPlatformInitializer.releaseExecutor.execute { CL.clReleaseMemObject(it) }
            }
        }
    }

    fun releaseKernel() = CL.clReleaseKernel(descriptor.kernel)

}

@KernelName("broadcast_to_shape", "broadcastTo")
class OCLBroadcastTo(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, target: OCLMemoryObject, newShape: List<Int>, oldShape: List<Int>) {
        runSingleDimKernel(
            newShape.toStoreSize().toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            newShape.size,
            newShape.toIntArray(),
            oldShape.toIntArray()
        )
    }
}

@KernelName("fill", "fill")
class OCLFill(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(target: OCLMemoryObject, range: IntRange, value: Float) {
        runSingleDimKernel(
            (range.last - range.first + 1L),
            target.getMemoryObject(SOURCE_AND_TARGET),
            range.first,
            value
        )
    }
}

@KernelName("elementwise_op_on_tensors", "elementwiseOpOnTensors")
class OCLElementwiseOpOnTensors(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source1: OCLMemoryObject, source2: OCLMemoryObject, target: OCLMemoryObject, op: OCLKernelConstant) {
        runSingleDimKernel(
            source1.size.toLong(),
            source1.getMemoryObject(SOURCE),
            source2.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            op.kernelConst,
            PlatformProvider.epsilon
        )
    }
}

@KernelName("elementwise_assign_op_on_tensors", "elementwiseAssignOpOnTensors")
class OCLElementwiseAssignOpOnTensors(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source1: OCLMemoryObject, source2: OCLMemoryObject, op: OCLKernelConstant) {
        runSingleDimKernel(
            source1.size.toLong(),
            source1.getMemoryObject(SOURCE_AND_TARGET),
            source2.getMemoryObject(SOURCE),
            op.kernelConst,
            PlatformProvider.epsilon
        )
    }
}

@KernelName("tensor_const_op", "tensorConstOp")
class OCLTensorConstOp(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, target: OCLMemoryObject, const: Float, op: OCLKernelConstant) {
        runSingleDimKernel(
            source.size.toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            const,
            op.kernelConst,
            PlatformProvider.epsilon
        )
    }
}

@KernelName("const_tensor_op", "constTensorOp")
class OCLConstTensorOp(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(const: Float, source: OCLMemoryObject, target: OCLMemoryObject, op: OCLKernelConstant) {
        runSingleDimKernel(
            source.size.toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            const,
            op.kernelConst,
            PlatformProvider.epsilon
        )
    }
}

@KernelName("tensor_const_assign_op", "tensorConstAssignOp")
class OCLTensorConstAssignOp(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, const: Float, op: OCLKernelConstant) {
        runSingleDimKernel(
            source.size.toLong(),
            source.getMemoryObject(SOURCE_AND_TARGET),
            const,
            op.kernelConst,
            PlatformProvider.epsilon
        )
    }
}

@KernelName("tensor_mapping_op", "tensorMappingOp")
class OCLTensorMappingOp(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, target: OCLMemoryObject, op: OCLKernelConstant) {
        runSingleDimKernel(
            source.size.toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            op.kernelConst
        )
    }
}

@KernelName("reduction_op", "reductionOp")
class OCLReductionOp(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, target: OCLMemoryObject, op: OCLKernelConstant) {
        runSingleDimKernel(
            target.size.toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            source.size,
            op.kernelConst
        )
    }
}

@KernelName("aggregate_over_axis", "aggregateOverAxis")
class OCLAggregateOverAxis(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, target: OCLMemoryObject, sourceShape: List<Int>, targetShape: List<Int>, axis: Int, op: OCLKernelConstant) {
        runSingleDimKernel(
            target.size.toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            targetShape.size,
            targetShape.toIntArray(),
            sourceShape.toIntArray(),
            axis,
            op.kernelConst
        )
    }
}

@KernelName("tensor_mapping_clamp", "clamp")
class OCLClamp(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, target: OCLMemoryObject, min: Float, max: Float) {
        runSingleDimKernel(
            source.size.toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            min, max
        )
    }
}

@KernelName("transpose", "transpose")
class OCLTranspose(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, target: OCLMemoryObject, oldShape: List<Int>, newShape: List<Int>, axis1: Int, axis2: Int) {
        runSingleDimKernel(
            source.size.toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            oldShape.toIntArray(),
            newShape.toIntArray(),
            oldShape.size,
            axis1, axis2
        )
    }
}

@KernelName("permute", "permute")
class OCLPermute(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject, target: OCLMemoryObject, oldShape: List<Int>, newShape: List<Int>, axes: List<Int>) {
        val axisPositions = IntArray(axes.size) { axes.indexOf(it) } //this allows to have a single indices array in the kernel
        runSingleDimKernel(
            source.size.toLong(),
            source.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            oldShape.toIntArray(),
            newShape.toIntArray(),
            axisPositions,
            oldShape.size
        )
    }
}

@KernelName("contains_nan", "containsNaN")
class OCLContainsNaN(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(source: OCLMemoryObject): Boolean {
        val res = AtomicBoolean(false)
        runSingleDimKernel(1L, source.getMemoryObject(SOURCE), res)
        return res.get()
    }
}

@KernelName("inplace_scatter", "inplaceScatter")
class OCLInplaceScatter(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(
        target: OCLMemoryObject,
        index: OCLMemoryObject,
        source: OCLMemoryObject,
        targetShape: List<Int>,
        indexShape: List<Int>,
        sourceShape: List<Int>,
        axis: Int
    ) {
        val indexShapeWithoutAxis = indexShape.toMutableList()
        indexShapeWithoutAxis.removeAt(axis)
        val invalidIndex = AtomicInteger(Int.MIN_VALUE)
        runSingleDimKernel(
            indexShapeWithoutAxis.toStoreSize().toLong(),
            target.getMemoryObject(SOURCE_AND_TARGET),
            index.getMemoryObject(SOURCE),
            source.getMemoryObject(SOURCE),
            targetShape.toIntArray(),
            indexShape.toIntArray(),
            sourceShape.toIntArray(),
            //invalidIndex,
            targetShape.size,
            axis
        )
        if (invalidIndex.get() != Int.MIN_VALUE) {
            throw IllegalArgumentException("Scatter got invalid index, index value ${invalidIndex.get()} " +
                    "is out of bounds 0 and ${targetShape[axis]} at axis: $axis")
        }
    }
}

@KernelName("inplace_scatter_add", "inplaceScatterAdd")
class OCLInplaceScatterAdd(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(
        target: OCLMemoryObject,
        index: OCLMemoryObject,
        source: OCLMemoryObject,
        targetShape: List<Int>,
        indexShape: List<Int>,
        sourceShape: List<Int>,
        axis: Int
    ) {
        val indexShapeWithoutAxis = indexShape.toMutableList()
        indexShapeWithoutAxis.removeAt(axis)
        val invalidIndex = AtomicInteger(Int.MIN_VALUE)
        runSingleDimKernel(
            1L,
            target.getMemoryObject(SOURCE_AND_TARGET),
            index.getMemoryObject(SOURCE),
            source.getMemoryObject(SOURCE),
            targetShape.toIntArray(),
            indexShape.toIntArray(),
            sourceShape.toIntArray(),
            //invalidIndex,
            targetShape.size,
            axis,
            indexShapeWithoutAxis.toStoreSize()
        )
        if (invalidIndex.get() != Int.MIN_VALUE) {
            throw IllegalArgumentException("ScatterAdd got invalid index, index value ${invalidIndex.get()} " +
                    "is out of bounds 0 and ${targetShape[axis]} at axis: $axis")
        }
    }
}

@KernelName("inplace_scatter_fill", "inplaceScatterFill")
class OCLInplaceScatterFill(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(
        target: OCLMemoryObject,
        index: OCLMemoryObject,
        targetShape: List<Int>,
        indexShape: List<Int>,
        axis: Int,
        value: Float
    ) {
        val indexShapeWithoutAxis = indexShape.toMutableList()
        indexShapeWithoutAxis.removeAt(axis)
        val invalidIndex = AtomicInteger(Int.MIN_VALUE)
        runSingleDimKernel(
            indexShapeWithoutAxis.toStoreSize().toLong(),
            target.getMemoryObject(SOURCE_AND_TARGET),
            index.getMemoryObject(SOURCE),
            targetShape.toIntArray(),
            indexShape.toIntArray(),
            //invalidIndex,
            targetShape.size,
            axis,
            value
        )
        if (invalidIndex.get() != Int.MIN_VALUE) {
            throw IllegalArgumentException("ScatterFill got invalid index, index value ${invalidIndex.get()} " +
                    "is out of bounds 0 and ${targetShape[axis]} at axis: $axis")
        }
    }
}

@KernelName("gather", "gather")
class OCLGather(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(
        source: OCLMemoryObject,
        index: OCLMemoryObject,
        target: OCLMemoryObject,
        sourceShape: List<Int>,
        indexShape: List<Int>,
        targetShape: List<Int>,
        axis: Int
    ) {
        val targetShapeWithoutAxis = targetShape.toMutableList()
        targetShapeWithoutAxis.removeAt(axis)
        val invalidIndex = AtomicInteger(Int.MIN_VALUE)
        runSingleDimKernel(
            targetShapeWithoutAxis.toStoreSize().toLong(),
            source.getMemoryObject(SOURCE),
            index.getMemoryObject(SOURCE),
            target.getMemoryObject(TARGET),
            sourceShape.toIntArray(),
            indexShape.toIntArray(),
            targetShape.toIntArray(),
            //invalidIndex,
            indexShape.size,
            axis
        )
        if (invalidIndex.get() != Int.MIN_VALUE) {
            throw IllegalArgumentException("Gather got invalid index, index value ${invalidIndex.get()} " +
                    "is out of bounds 0 and ${targetShape[axis]} at axis: $axis")
        }
    }
}

@KernelName("fill_random", "fillRandom")
class OCLFillRandom(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(target: OCLMemoryObject) {
        runSingleDimKernel(
            target.size.toLong(),
            target.getMemoryObject(TARGET),
            Random.nextLong(),
            OCLKernelConstant.NORMAL.kernelConst,
            0.0f
        )
    }
}

@KernelName("fill_random", "fillBernoulli")
class OCLFillBernoulli(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(target: OCLMemoryObject, rate: Float) {
        runSingleDimKernel(
            target.size.toLong(),
            Random.nextLong(),
            OCLKernelConstant.BERNOULLI.kernelConst,
            rate
        )
    }
}

@KernelName("col2im_for_transpose", "col2imForTranspose")
class OCLCol2ImForTranspose(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(
        col: OCLMemoryObject,
        kernelHeight: Int, kernelWidth: Int,
        paddingHeight: Int, paddingWidth: Int,
        strideHeight: Int, strideWidth: Int,
        dilationHeight: Int, dilationWidth: Int,
        channels: Int,
        height: Int, width: Int,
        outputHeight: Int, outputWidth: Int,
        im: OCLStoreView
    ) {
        val colChannels = channels * kernelHeight * kernelWidth
        runSingleDimKernel(
            (outputHeight * outputWidth).toLong(),
            col.getMemoryObject(SOURCE),
            kernelHeight, kernelWidth,
            paddingHeight, paddingWidth,
            strideHeight, strideWidth,
            dilationHeight, dilationWidth,
            colChannels,
            height, width,
            outputHeight, outputWidth,
            im.getMemoryObject(SOURCE_AND_TARGET),
            im.offset
        )
    }
}

@KernelName("vol2col", "vol2col")
class OCLVol2Col(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(
        vol: OCLStoreView,
        kernelDepth: Int, kernelHeight: Int, kernelWidth: Int,
        paddingDepth: Int, paddingHeight: Int, paddingWidth: Int,
        strideDepth: Int, strideHeight: Int, strideWidth: Int,
        dilationDepth: Int, dilationHeight: Int, dilationWidth: Int,
        channels: Int,
        depth: Int, height: Int, width: Int,
        col: OCLMemoryObject
    ) {
        val colChannels = channels * kernelDepth * kernelHeight * kernelWidth;
        val outputDepth = (depth + 2 * paddingDepth - (dilationDepth * (kernelDepth -1) + 1)) / strideDepth + 1
        val outputHeight = (height + 2 * paddingHeight - (dilationHeight * (kernelHeight - 1) + 1)) / strideHeight + 1
        val outputWidth = (width + 2 * paddingWidth - (dilationWidth * (kernelWidth - 1) + 1)) / strideWidth + 1
        runSingleDimKernel(
            (outputDepth * outputHeight * outputWidth).toLong(),
            vol.getMemoryObject(SOURCE), vol.offset,
            kernelDepth, kernelHeight, kernelWidth,
            paddingDepth, paddingHeight, paddingWidth,
            strideDepth, strideHeight, strideWidth,
            dilationDepth, dilationHeight, dilationWidth,
            colChannels,
            depth, height, width,
            outputDepth, outputHeight, outputWidth,
            col.getMemoryObject(TARGET)
        )
    }
}

@KernelName("col2vol", "col2vol")
class OCLCol2Vol(descriptor: OCLKernelDescriptor): AbstractOCLKernel(descriptor) {
    operator fun invoke(
        col: OCLMemoryObject,
        kernelDepth: Int, kernelHeight: Int, kernelWidth: Int,
        paddingDepth: Int, paddingHeight: Int, paddingWidth: Int,
        strideDepth: Int, strideHeight: Int, strideWidth: Int,
        dilationDepth: Int, dilationHeight: Int, dilationWidth: Int,
        channels: Int,
        depth: Int, height: Int, width: Int,
        outputDepth: Int, outputHeight: Int, outputWidth: Int,
        vol: OCLStoreView
    ) {
        val colChannels = channels * kernelDepth * kernelHeight * kernelWidth;
        runSingleDimKernel(
            (outputDepth * outputHeight * outputWidth).toLong(),
            col.getMemoryObject(SOURCE),
            kernelDepth, kernelHeight, kernelWidth,
            paddingDepth, paddingHeight, paddingWidth,
            strideDepth, strideHeight, strideWidth,
            dilationDepth, dilationHeight, dilationWidth,
            colChannels,
            depth, height, width,
            outputDepth, outputHeight, outputWidth,
            vol.getMemoryObject(SOURCE_AND_TARGET), vol.offset
        )
    }
}

