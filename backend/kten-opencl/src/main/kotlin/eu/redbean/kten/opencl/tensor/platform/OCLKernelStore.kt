package eu.redbean.kten.opencl.tensor.platform

import eu.redbean.kten.opencl.tensor.platform.kernels.*
import org.jocl.*
import kotlin.reflect.full.findAnnotation

class OCLKernelStore(
    private val context: cl_context,
    private val commandQueue: cl_command_queue,
    private val platformSpecInfo: OCLPlatformSpecInfo
) {

    private val program: cl_program
    private val kernels = mutableMapOf<String, AbstractOCLKernel>()

    val broadcastTo: OCLBroadcastTo by kernels
    val fill: OCLFill by kernels
    val elementwiseOpOnTensors: OCLElementwiseOpOnTensors by kernels
    val elementwiseAssignOpOnTensors: OCLElementwiseAssignOpOnTensors by kernels
    val tensorConstOp: OCLTensorConstOp by kernels
    val constTensorOp: OCLConstTensorOp by kernels
    val tensorConstAssignOp: OCLTensorConstAssignOp by kernels
    val tensorMappingOp: OCLTensorMappingOp by kernels
    val reductionOp: OCLReductionOp by kernels
    val aggregateOverAxis: OCLAggregateOverAxis by kernels
    val clamp: OCLClamp by kernels
    val transpose: OCLTranspose by kernels
    val permute: OCLPermute by kernels
    val containsNaN: OCLContainsNaN by kernels
    val inplaceScatter: OCLInplaceScatter by kernels
    val inplaceScatterAdd: OCLInplaceScatterAdd by kernels
    val inplaceScatterFill: OCLInplaceScatterFill by kernels
    val gather: OCLGather by kernels
    val fillRandom: OCLFillRandom by kernels
    val fillBernoulli: OCLFillBernoulli by kernels
    val col2imForTranspose: OCLCol2ImForTranspose by kernels
    val vol2col: OCLVol2Col by kernels
    val col2vol: OCLCol2Vol by kernels
    val maxPoolingUpdateOutput: OCLMaxPoolUpdateOut by kernels
    val maxPoolingUpdateGradIn: OCLMaxPoolUpdateGradIn by kernels
    val avgPoolingUpdateOutput: OCLAvgPoolUpdateOut by kernels
    val avgPoolingUpdateGradIn: OCLAvgPoolUpdateGradIn by kernels
    val batchNormUpdateOutput: OCLBatchNormUpdateOutput by kernels
    val batchNormUpdateGrads: OCLBatchNormUpdateGrads by kernels
    val indexSelect: OCLIndexSelect by kernels
    val indexAdd: OCLIndexAdd by kernels
    val upsampleNearestUpdateOutput: OCLUpsampleNearestUpdateOutput by kernels
    val upsampleNearestUpdateGrad: OCLUpsampleNearestUpdateGrad by kernels
    val maskedFill: OCLMaskedFill by kernels

    init {
        val source = this.javaClass.getResource("/kernels.cl").readText()
        program = CL.clCreateProgramWithSource(
            context,
            1,
            Array(1) { source },
            null, null
        )
        CL.clBuildProgram(program, 0, null, null, null, null)
        kernels += create(::OCLBroadcastTo)
        kernels += create(::OCLFill)
        kernels += create(::OCLElementwiseOpOnTensors)
        kernels += create(::OCLElementwiseAssignOpOnTensors)
        kernels += create(::OCLTensorConstOp)
        kernels += create(::OCLConstTensorOp)
        kernels += create(::OCLTensorConstAssignOp)
        kernels += create(::OCLTensorMappingOp)
        kernels += create(::OCLReductionOp)
        kernels += create(::OCLAggregateOverAxis)
        kernels += create(::OCLClamp)
        kernels += create(::OCLTranspose)
        kernels += create(::OCLPermute)
        kernels += create(::OCLContainsNaN)
        kernels += create(::OCLInplaceScatter)
        kernels += create(::OCLInplaceScatterAdd)
        kernels += create(::OCLInplaceScatterFill)
        kernels += create(::OCLGather)
        kernels += create(::OCLFillRandom)
        kernels += create(::OCLFillBernoulli)
        kernels += create(::OCLCol2ImForTranspose)
        kernels += create(::OCLVol2Col)
        kernels += create(::OCLCol2Vol)
        kernels += create(::OCLMaxPoolUpdateOut)
        kernels += create(::OCLMaxPoolUpdateGradIn)
        kernels += create(::OCLAvgPoolUpdateOut)
        kernels += create(::OCLAvgPoolUpdateGradIn)
        kernels += create(::OCLBatchNormUpdateOutput)
        kernels += create(::OCLBatchNormUpdateGrads)
        kernels += create(::OCLIndexSelect)
        kernels += create(::OCLIndexAdd)
        kernels += create(::OCLUpsampleNearestUpdateOutput)
        kernels += create(::OCLUpsampleNearestUpdateGrad)
        kernels += create(::OCLMaskedFill)
    }

    private inline fun <reified T: AbstractOCLKernel> create(kernel: (OCLKernelDescriptor) -> T): Pair<String, T> {
        val annotation = T::class.findAnnotation<KernelName>()!!
        return annotation.kotlinName to kernel(createDescriptor(annotation.functionName))
    }

    private fun createDescriptor(kernelName: String): OCLKernelDescriptor = OCLKernelDescriptor(
        CL.clCreateKernel(program, kernelName, null),
        context,
        commandQueue,
        program,
        platformSpecInfo
    )

    fun releaseAll() {
        kernels.values.forEach(AbstractOCLKernel::releaseKernel)
        CL.clReleaseProgram(program)
    }

}