#define MAX_DIMENSIONS 10
#define OP_PLUS 0
#define OP_MINUS 1
#define OP_TIMES 2
#define OP_DIV 3
#define OP_POW 4
#define OP_RECIPROCAL 5
#define OP_LOG 6
#define OP_SUM 7
#define OP_MEAN 8
#define OP_MAX 9
#define OP_MIN 10
#define OP_ARG_MAX 11
#define OP_ARG_MIN 12
#define OP_LT 13
#define OP_LTE 14
#define OP_GT 15
#define OP_GTE 16
#define OP_EQ 17
#define OP_NEQ 18
#define OP_EXP 19
#define OP_TANH 20
#define OP_SIGMOID 21
#define OP_SINH 22
#define OP_COSH 23
#define OP_ABS 24
#define OP_SIGN 25
#define OP_SQRT 26
#define OP_SIN 27
#define OP_COS 28
#define OP_TAN 29
#define OP_ASIN 30
#define OP_ACOS 31
#define OP_ATAN 32
#define OP_FLOOR 33
#define OP_CEIL 34
#define OP_ROUND 35
#define OP_TRUNC 36
#define OP_RSQRT 37
#define NORMAL 38
#define BERNOULLI 39


__kernel void broadcast_to_shape(__global float *t1,
                                 __global float *res,
                                 const int size,
                                 __global const int *shape,
                                 __global const int *origShape)
{
    int gid = get_global_id(0);

    uint indices[MAX_DIMENSIONS];

    int i = gid;

    for (int j = size - 1; j >= 0; j--) {
        int dim = shape[j];
        int subIdx = i % dim;
        indices[j] = subIdx % origShape[j];
        i = i / dim;
    }

    int realIdx = 0;

    for (int k = 0; k < size; k++) {
        realIdx = realIdx * origShape[k] + indices[k];
    }

    res[gid] = t1[realIdx];
}


__kernel void fill(__global float *tensor, const int offset, const float value) {
    int gid = get_global_id(0);
    tensor[gid + offset] = value;
}


float calc_by_elwise_op(float a, float b, int op, float epsilon) {
    switch(op) {
        case OP_PLUS: return a + b;
        case OP_MINUS: return a - b;
        case OP_TIMES: return a * b;
        case OP_DIV: return a / b;
        case OP_POW: return pow(a, b);
        case OP_LT: return a < b;
        case OP_LTE: return a <= b;
        case OP_GT: return a > b;
        case OP_GTE: return a >= b;
        case OP_EQ: return fabs(a - b) < epsilon;
        case OP_NEQ: return fabs(a - b) > epsilon;
        default: return NAN;
    }
}

__kernel void elementwise_op_on_tensors(__global float *t1, __global float *t2, __global float *res, const int op, const float epsilon) {
    int gid = get_global_id(0);
    res[gid] = calc_by_elwise_op(t1[gid], t2[gid], op, epsilon);
}


__kernel void elementwise_assign_op_on_tensors(__global float *t1, __global float *t2, const int op, const float epsilon) {
    int gid = get_global_id(0);
    t1[gid] = calc_by_elwise_op(t1[gid], t2[gid], op, epsilon);
}


__kernel void tensor_const_op(__global float *tensor, __global float *res, const float value, const int op, const float epsilon) {
    int gid = get_global_id(0);
    res[gid] = calc_by_elwise_op(tensor[gid], value, op, epsilon);
}


__kernel void const_tensor_op(__global float *tensor, __global float *res, const float value, const int op, const float epsilon) {
    int gid = get_global_id(0);
    res[gid] = calc_by_elwise_op(value, tensor[gid], op, epsilon);
}


__kernel void tensor_const_assign_op(__global float *tensor, const float value, const int op, const float epsilon) {
    int gid = get_global_id(0);
    tensor[gid] = calc_by_elwise_op(tensor[gid], value, op, epsilon);
}


__kernel void tensor_mapping_op(__global float *tensor, __global float *res, const int op) {
    int gid = get_global_id(0);
    switch(op) {
        case OP_RECIPROCAL: res[gid] = 1.0f / tensor[gid]; break;
        case OP_LOG: res[gid] = log(tensor[gid]); break;
        case OP_EXP: res[gid] = exp(tensor[gid]); break;
        case OP_TANH: res[gid] = tanh(tensor[gid]); break;
        case OP_SIGMOID: res[gid] = 1.0f / (1.0f + exp(-1.0f * tensor[gid])); break;
        case OP_SIGN: res[gid] = sign(tensor[gid]); break;
        case OP_SQRT: res[gid] = sqrt(tensor[gid]); break;
        case OP_SIN: res[gid] = sin(tensor[gid]); break;
        case OP_COS: res[gid] = cos(tensor[gid]); break;
        case OP_TAN: res[gid] = tan(tensor[gid]); break;
        case OP_ASIN: res[gid] = asin(tensor[gid]); break;
        case OP_ACOS: res[gid] = acos(tensor[gid]); break;
        case OP_ATAN: res[gid] = atan(tensor[gid]); break;
        case OP_FLOOR: res[gid] = floor(tensor[gid]); break;
        case OP_CEIL: res[gid] = ceil(tensor[gid]); break;
        case OP_ROUND: res[gid] = round(tensor[gid]); break;
        case OP_TRUNC: res[gid] = trunc(tensor[gid]); break;
        case OP_RSQRT: res[gid] = rsqrt(tensor[gid]); break;
        case OP_ABS: res[gid] = fabs(tensor[gid]); break;
        default: res[gid] = NAN;
    }
}


__kernel void tensor_mapping_clamp(__global float *tensor, __global float *res, const float min, const float max) {
    int gid = get_global_id(0);
    res[gid] = clamp(tensor[gid], min, max);
}


__kernel void reduction_op(__global float *tensor, __global float *res, const int elements, const int elementsOrig, const int op) {
    int gid = get_global_id(0);
    res[gid] = 0.0f;
    for (int i = 0; i < elements; i++) {
        res[gid] += tensor[i];
    }
    if (op == OP_MEAN)
        res[gid] = res[gid] / (float) elementsOrig;
}


int single_index(int *indices, int size, __global int *shape) {
    int realIdx = 0;

    for (int i = 0; i < size; i++) {
        realIdx = realIdx * shape[i] + indices[i];
    }

    return realIdx;
}

float calc_agg_op(int operation, float aggVal, float currentVal) {
    if (operation == OP_SUM) {
        return aggVal + currentVal;
    } else if (operation == OP_MAX) {
        return fmax(aggVal, currentVal);
    } else if (operation == OP_MIN) {
        return fmin(aggVal, currentVal);
    } else {
        return NAN;
    }
}

__kernel void aggregate_over_axis(__global float *t1,
                                  __global float *res,
                                  const int size,
                                  __global const int *shape,
                                  __global const int *origShape,
                                  const int axis,
                                  const int operation)
{
    int gid = get_global_id(0);

    int indices[MAX_DIMENSIONS];

    int i = gid;

    for (int j = size - 1; j >= 0; j--) {
        int dim = shape[j];
        indices[j] = i % dim;
        i = i / dim;
    }

    float aggVal = 0.0f;

    if (operation == OP_MAX) {
        aggVal = -MAXFLOAT;
    } else if (operation == OP_MIN) {
        aggVal = MAXFLOAT;
    }

    float temp;

    if (operation == OP_ARG_MAX) {
        temp = -MAXFLOAT;
    } else if (operation == OP_ARG_MIN) {
        temp = MAXFLOAT;
    }

    for (int k = 0; k < origShape[axis]; k++) {
        indices[axis] = k;
        float currentVal = t1[single_index(indices, size, origShape)];

        if (operation == OP_ARG_MAX) {
            if (currentVal > temp) {
                temp = currentVal;
                aggVal = k;
            }
        } else if (operation == OP_ARG_MIN) {
            if (currentVal < temp) {
                temp = currentVal;
                aggVal = k;
            }
        } else if (operation == OP_MEAN) {
            aggVal = calc_agg_op(OP_SUM, aggVal, currentVal);
        } else {
            aggVal = calc_agg_op(operation, aggVal, currentVal);
        }
    }

    if (operation == OP_MEAN) {
        aggVal = aggVal / ((float) origShape[axis]);
    }

    res[gid] = aggVal;
}


__kernel void transpose(__global float *tensor,
                        __global float *res,
                        __global int *shape,
                        __global int *newShape,
                        const int size,
                        const int axis1,
                        const int axis2) {
    int gid = get_global_id(0);

    int indices[MAX_DIMENSIONS];

    int i = gid;

    for (int j = size - 1; j >= 0; j--) {
        int dim = shape[j];
        indices[j] = i % dim;
        i = i / dim;
    }

    int temp = indices[axis1];
    indices[axis1] = indices[axis2];
    indices[axis2] = temp;

    res[single_index(indices, size, newShape)] = tensor[gid];
}


__kernel void permute(__global float *tensor,
                      __global float *res,
                      __global int *shape,
                      __global int *newShape,
                      __global int *axisPositions,
                      const int dimensions) {
    int gid = get_global_id(0);
    int indices[MAX_DIMENSIONS];

    int i = gid;

    for (int j = dimensions - 1; j >= 0; j--) {
        int dim = shape[j];
        indices[axisPositions[j]] = i % dim;
        i = i / dim;
    }

    res[single_index(indices, dimensions, newShape)] = tensor[gid];
}


__kernel void contains_nan(__global float *tensor, __global int *res, const int tensor_size) {
    for (int i = 0; i < tensor_size; i++) {
        if (isnan(tensor[i])) {
            res[0] = 1;
            return;
        }
    }
    res[0] = 0;
}


#define APPLY_AT_AXIS(AXIS, ELEMENT_INDEX, DIMENSIONS, SHAPE, CODE) \
{ \
    int indices[MAX_DIMENSIONS]; \
    for (int j = DIMENSIONS - 1; j >= 0; j--) { \
        if (j != AXIS) { \
            int dim = SHAPE[j]; \
            indices[j] = ELEMENT_INDEX % dim; \
            ELEMENT_INDEX = ELEMENT_INDEX / dim; \
        } \
    } \
    CODE \
}

int index_mapping(int *indices, int dimensions, int axis, int idx, __global int *shape) {
    int realIdx = 0;

    for (int i = 0; i < dimensions; i++) {
        if (i == axis) {
            realIdx = realIdx * shape[i] + idx;
        } else {
            realIdx = realIdx * shape[i] + indices[i];
        }
    }

    return realIdx;
}

#define APPLY_FOR_AXIS(AXIS, ELEMENT_SIZE, ELEMENT_INDEX_IN_AXIS, DIMENSIONS, SHAPE, CODE) \
{ \
    int offset = 1; \
    for (int shapeIndex = AXIS + 1; shapeIndex < DIMENSIONS; shapeIndex++) { \
        offset *= SHAPE[shapeIndex]; \
    } \
    for (int counter = 0; counter < ELEMENT_SIZE; counter++) { \
        int preset = counter / offset; \
        int realIndex = counter % offset; \
        realIndex = preset * SHAPE[AXIS] * offset + ELEMENT_INDEX_IN_AXIS * offset + realIndex; \
        CODE \
    } \
}

__kernel void batch_norm_update_out(
                                    __global float *input,
                                    __global float *output,
                                    __global float *gamma,
                                    __global float *beta,
                                    __global float *runningMean,
                                    __global float *runningVar,
                                    __global float *currentMean,
                                    __global float *currentStd,
                                    __global int *inputShape,
                                    const int dimensions,
                                    const int inputElementSize,
                                    const int axis,
                                    const float epsilon,
                                    const float momentum,
                                    const int training
) {
    int gid = get_global_id(0);

    int elementSizeAtSingleIndexForAxis = inputElementSize / inputShape[axis];

    float mean;
    float stdev = 0.0;

    if (training == 1) {
        float sum = 0.0;

        APPLY_FOR_AXIS(axis, elementSizeAtSingleIndexForAxis, gid, dimensions, inputShape, sum += input[realIndex];)

        mean = sum / elementSizeAtSingleIndexForAxis;

        currentMean[gid] = mean;

        sum = 0.0;
        APPLY_FOR_AXIS(axis, elementSizeAtSingleIndexForAxis, gid, dimensions, inputShape, {
            float inputMinusMean = input[realIndex] - mean;
            sum += inputMinusMean * inputMinusMean;
        })

        if (sum != 0.0 || epsilon != 0.0) {
            stdev = (float) (1.0 / sqrt(sum / elementSizeAtSingleIndexForAxis + epsilon));
        }

        currentStd[gid] = stdev;

        if (runningMean && runningVar) {
            runningMean[gid] = (momentum * mean) + ((1.0 - momentum) * runningMean[gid]);
            float unbiasedVar = sum / (elementSizeAtSingleIndexForAxis - 1);
            runningVar[gid] = (momentum * unbiasedVar) + ((1.0 - momentum) * runningVar[gid]);
        }
    } else {
        mean = runningMean[gid];
        stdev = 1.0 / sqrt(runningVar[gid] + epsilon);
    }

    float g = (gamma) ? gamma[gid] : 1.0;
    float b = (beta) ? beta[gid] : 0.0;

    APPLY_FOR_AXIS(axis, elementSizeAtSingleIndexForAxis, gid, dimensions, inputShape, {
        output[realIndex] = (input[realIndex] - mean) * stdev * g + b;
    })
}

__kernel void batch_norm_update_grads(
                                __global float *input,
                                __global float *gradOut,
                                __global float *gamma,
                                __global float *runningMean,
                                __global float *runningVar,
                                __global float *currentMean,
                                __global float *currentStd,
                                __global float *gradIn,
                                __global float *gradGamma,
                                __global float *gradBeta,
                                __global int *inputShape,
                                const int dimensions,
                                const int inputElementSize,
                                const int axis,
                                const float epsilon,
                                const int training
) {
    int gid = get_global_id(0);

    int elementSizeAtSingleIndexForAxis = inputElementSize / inputShape[axis];

    float mean;
    float stdev;

    float gamma_at_gid = (gamma) ? gamma[gid] : 1.0;

    if (training == 1) {
        mean = currentMean[gid];
        stdev = currentStd[gid];
    } else {
        mean = runningMean[gid];
        stdev = 1.0 / sqrt(runningVar[gid] + epsilon);
    }

    float sum = 0.0;
    APPLY_FOR_AXIS(axis, elementSizeAtSingleIndexForAxis, gid, dimensions, inputShape,
        sum += gradOut[realIndex];
    )

    float dotProd = 0.0;
    APPLY_FOR_AXIS(axis, elementSizeAtSingleIndexForAxis, gid, dimensions, inputShape, {
        dotProd += (input[realIndex] - mean) * gradOut[realIndex];
    })

    if (gradIn) {
        if (training == 1) {
            float dotpStdSq = dotProd * stdev * stdev / elementSizeAtSingleIndexForAxis;
            float gradMean = sum / elementSizeAtSingleIndexForAxis;
            APPLY_FOR_AXIS(axis, elementSizeAtSingleIndexForAxis, gid, dimensions, inputShape, {
                float gradInBase = (input[realIndex] - mean) * dotpStdSq;
                gradIn[realIndex] = (gradOut[realIndex] - gradMean - gradInBase) * stdev * gamma_at_gid;
            })
        } else {
            APPLY_FOR_AXIS(axis, elementSizeAtSingleIndexForAxis, gid, dimensions, inputShape, {
                gradIn[realIndex] = gradOut[realIndex] * stdev * gamma_at_gid;
            })
        }
    }

    if (gradGamma) {
        gradGamma[gid] = dotProd * stdev;
    }

    if (gradBeta) {
        gradBeta[gid] = sum;
    }
}

__kernel void gather(__global float *tensor,
                     __global float *index,
                     __global float *res,
                     __global int *tensor_shape,
                     __global int *index_shape,
                     __global int *res_shape,
                     //__global int *invalid_index,
                     const int dimensions,
                     const int axis) {
    int gid = get_global_id(0);

    int i = gid;

    APPLY_AT_AXIS(axis, i, dimensions, res_shape,
        for (int k = 0; k < index_shape[axis]; k++) {
            int idx = index[index_mapping(indices, dimensions, axis, k, index_shape)];
            if (idx < 0 || idx >= tensor_shape[axis]) {
                //invalid_index[0] = idx;
                res[0] = NAN;
                return;
            }
            res[index_mapping(indices, dimensions, axis, k, res_shape)] = tensor[index_mapping(indices, dimensions, axis, idx, tensor_shape)];
        })
}


__kernel void inplace_scatter(__global float *tensor,
                              __global float *index,
                              __global float *source,
                              __global int *tensor_shape,
                              __global int *index_shape,
                              __global int *source_shape,
                              //__global int *invalid_index,
                              const int dimensions,
                              const int axis) {

    int gid = get_global_id(0);
    int i = gid;

    APPLY_AT_AXIS(axis, i, dimensions, index_shape,
        for (int k = 0; k < index_shape[axis]; k++) {
            int idx = index[index_mapping(indices, dimensions, axis, k, index_shape)];
            if (idx < 0 || idx >= tensor_shape[axis]) {
                //invalid_index[0] = idx;
                tensor[0] = NAN;
                return;
            }
            tensor[index_mapping(indices, dimensions, axis, idx, tensor_shape)] = source[index_mapping(indices, dimensions, axis, k, source_shape)];
        })

}


__kernel void inplace_scatter_add(__global float *tensor,
                                  __global float *index,
                                  __global float *source,
                                  __global int *tensor_shape,
                                  __global int *index_shape,
                                  __global int *source_shape,
                                  //__global int *invalid_index,
                                  const int dimensions,
                                  const int axis,
                                  const int index_size) {

    for(int i = 0; i < index_size; i++) {
        int i_mutable = i;
        APPLY_AT_AXIS(axis, i_mutable, dimensions, index_shape,
          for (int k = 0; k < index_shape[axis]; k++) {
              int idx = index[index_mapping(indices, dimensions, axis, k, index_shape)];
              if (idx < 0 || idx >= tensor_shape[axis]) {
                  //invalid_index[0] = idx;
                  tensor[0] = NAN;
                  return;
              }
              float acc = tensor[index_mapping(indices, dimensions, axis, idx, tensor_shape)];
              tensor[index_mapping(indices, dimensions, axis, idx, tensor_shape)] = acc + source[index_mapping(indices, dimensions, axis, k, source_shape)];
          })
    }
}


__kernel void inplace_scatter_fill(__global float *tensor,
                                   __global float *index,
                                   __global int *tensor_shape,
                                   __global int *index_shape,
                                   //__global int *invalid_index,
                                   const int dimensions,
                                   const int axis,
                                   const float value) {
    int gid = get_global_id(0);
    int i = gid;

    APPLY_AT_AXIS(axis, i, dimensions, index_shape,
        for (int k = 0; k < index_shape[axis]; k++) {
            int idx = index[index_mapping(indices, dimensions, axis, k, index_shape)];
            if (idx < 0 || idx >= tensor_shape[axis]) {
                //invalid_index[0] = idx;
                tensor[0] = NAN;
                return;
            }
            tensor[index_mapping(indices, dimensions, axis, idx, tensor_shape)] = value;
        })
}


__kernel void index_select(__global float *tensor,
                           __global float *index,
                           __global float *output,
                           __global int *tensor_shape,
                           __global int *output_shape,
                           const int dimensions,
                           const int index_size,
                           const int tensor_size_at_axis,
                           const int axis
) {
    int gid = get_global_id(0);
    int i = gid;

    APPLY_AT_AXIS(axis, i, dimensions, output_shape,
        for (int k = 0; k < index_size; k++) {
            int idx = index[k];
            if (idx < 0 || idx >= tensor_size_at_axis) {
                output[0] = NAN;
                return;
            }
            output[index_mapping(indices, dimensions, axis, k, output_shape)] = tensor[index_mapping(indices, dimensions, axis, idx, tensor_shape)];
        }
    )
}


__kernel void index_add(__global float *tensor,
                        __global float *index,
                        __global float *source,
                        __global int *tensor_shape,
                        __global int *source_shape,
                        const int dimensions,
                        const int index_size,
                        const int tensor_size_at_axis,
                        const int axis,
                        const float alpha
) {
    int gid = get_global_id(0);
    int i = gid;

    APPLY_AT_AXIS(axis, i, dimensions, tensor_shape,
        for (int k = 0; k < index_size; k++) {
            int idx = index[k];
            if (idx < 0 || idx >= tensor_size_at_axis) {
                tensor[0] = NAN;
                return;
            }
            tensor[index_mapping(indices, dimensions, axis, idx, tensor_shape)] += source[index_mapping(indices, dimensions, axis, k, source_shape)] * alpha;
        }
    )
}


int next(int bits, ulong seed) {
    seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
    return seed >> (48 - bits);
}

float next_float(ulong seed) {
    return next(24, seed) / ((float)(1 << 24));
}

float next_gauss(ulong seed, int gid) {
    int cnt = 0;
    float v1, v2, s;
    do {
        if (cnt > 100) {
            break;
        }
        v1 = 2.0f * next_float(seed) - 1.0f;
        v2 = 2.0f * next_float(seed) - 1.0f;
        s = v1 * v1 + v2 * v2;
        seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
        cnt++;
    } while(s >= 1.0f || s == 0.0f);

    if (cnt > 100)
        return 0.0f; // no way to indicate error

    float multiplier = sqrt(-2 * log(s) / s);
    if (gid % 2 == 0) {
        return v1 * multiplier;
    } else {
        return v2 * multiplier;
    }
}

__kernel void fill_random(__global float *res,
                         const long base,
                         const int mode,
                         const float rate)
{
    int gid = get_global_id(0);

    ulong seed = gid * base + ((ulong) (res[gid] * base));

    if (mode == NORMAL) {
        res[gid] = next_gauss(seed, gid);
    } else {
        res[gid] = next_float(seed) < rate ? 1.0f : 0.0f;
    }
}

__kernel void col2im_for_transpose(__global float *col,
                                   const int kernelH, const int kernelW,
                                   const int paddingH, const int paddingW,
                                   const int strideH, const int strideW,
                                   const int dilationH, const int dilationW,
                                   const int colChannels,
                                   const int inputH, const int inputW,
                                   const int outputH, const int outputW,
                                   __global float *im, const long imOffset)
{
    int gid = get_global_id(0); // outputH * outputW
    int colW = gid % outputW;
    int colH = (gid / outputW) % outputH;

    for (int colC = 0; colC < colChannels; colC++) {
        int offsetW = colC % kernelW;
        int offsetH = (colC / kernelW) % kernelH;
        int imC = colC / kernelH / kernelW;

        int imH = colH * strideH - paddingH + offsetH * dilationH;
        int imW = colW * strideW - paddingW + offsetW * dilationW;

        if ((imH >= 0 && imH < inputH) && (imW >= 0 && imW < inputW)) {
            im[imOffset + (imC * inputH + imH) * inputW + imW] += col[(colC * outputH + colH) * outputW + colW];
        }
    }
}


__kernel void vol2col(__global float *vol, const long volOffset,
                      const int kernelD, const int kernelH, const int kernelW,
                      const int paddingD, const int paddingH, const int paddingW,
                      const int strideD, const int strideH, const int strideW,
                      const int dilationD, const int dilationH, const int dilationW,
                      const int colChannels,
                      const int inputD, const int inputH, const int inputW,
                      const int outputD, const int outputH, const int outputW,
                      __global float *col)
{
    int gid = get_global_id(0); // outputD * outputH * outputW
    int colW = gid % outputW;
    int colH = (gid / outputW) % outputH;
    int colD = gid / outputW / outputH;

    for (int colC = 0; colC < colChannels; colC++) {
        int offsetW = colC % kernelW;
        int offsetH = (colC / kernelW) % kernelH;
        int offsetD = (colC / kernelW / kernelH) % kernelD;
        int volC = colC / kernelD / kernelH / kernelW;

        int volD = colD * strideD - paddingD + offsetD + dilationD;
        int volH = colH * strideH - paddingH + offsetH * dilationH;
        int volW = colW * strideW - paddingW + offsetW * dilationW;

        float volValue = 0.0f;
        if ((volD >= 0 && volD < inputD) && (volH >= 0 && volH < inputH) && (volW >= 0 && volW < inputW)) {
            volValue = vol[((volC * inputD + volD) * inputH + volH) * inputW + volW + volOffset];
        }

        col[((colC * outputD + colD) * outputH + colH) * outputW + colW] = volValue;
    }
}


__kernel void col2vol(__global float *col,
                      const int kernelD, const int kernelH, const int kernelW,
                      const int paddingD, const int paddingH, const int paddingW,
                      const int strideD, const int strideH, const int strideW,
                      const int dilationD, const int dilationH, const int dilationW,
                      const int colChannels,
                      const int inputD, const int inputH, const int inputW,
                      const int outputD, const int outputH, const int outputW,
                      __global float *vol, const long volOffset)
{
    int gid = get_global_id(0); // outputD * outputH * outputW
    int colW = gid % outputW;
    int colH = (gid / outputW) % outputH;
    int colD = gid / outputW / outputH;

    for (int colC = 0; colC < colChannels; colC++) {
        int offsetW = colC % kernelW;
        int offsetH = (colC / kernelW) % kernelH;
        int offsetD = (colC / kernelW / kernelH) % kernelD;
        int volC = colC / kernelD / kernelH / kernelW;

        int volD = colD * strideD - paddingD + offsetD + dilationD;
        int volH = colH * strideH - paddingH + offsetH * dilationH;
        int volW = colW * strideW - paddingW + offsetW * dilationW;

        vol[((volC * inputD + volD) * inputH + volH) * inputW + volW + volOffset] += col[((colC * outputD + colD) * outputH + colH) * outputW + colW];
    }
}

__kernel void max_pool_update_out(__global float *input,
                                  __global float *output,
                                  __global float *indices,
                                  const int inputHeight, const int inputWidth,
                                  const int outputHeight, const int outputWidth,
                                  const int kernelH, const int kernelW,
                                  const int strideH, const int strideW,
                                  const int paddingH, const int paddingW,
                                  const int dilationH, const int dilationW)
{
    int gid = get_global_id(0); // [batch] * inputPlane
    int inputStartPos = gid * inputHeight * inputWidth;

    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            int hStart = i * strideH - paddingH;
            int wStart = j * strideW - paddingW;
            int hEnd = min(hStart + (kernelH - 1) * dilationH + 1, inputHeight);
            int wEnd = min(wStart + (kernelW - 1) * dilationW + 1, inputWidth);
            while (hStart < 0) hStart += dilationH;
            while (wStart < 0) wStart += dilationW;

            int outputStartPos = gid * outputHeight * outputWidth + i * outputWidth + j;

            float maxIndex = -1.0;
            float maxVal = -MAXFLOAT;

            for (int y = hStart; y < hEnd; y += dilationH) {
                for (int x = wStart; x < wEnd; x += dilationW) {
                    int idx = y * inputWidth + x;
                    float val = input[inputStartPos + idx];
                    if (val > maxVal) {
                        maxVal = val;
                        maxIndex = (float) idx;
                    }
                }
            }

            output[outputStartPos] = maxVal;
            indices[outputStartPos] = maxIndex;
        }
    }
}

__kernel void max_pool_update_grad_in(__global float *gradIn,
                                      __global float *gradOut,
                                      __global float *indices,
                                      const int inputHeight, const int inputWidth,
                                      const int outputHeight, const int outputWidth)
{
    int gid = get_global_id(0); // [batch] * inputPlane
    int gradInStartPos = gid * inputHeight * inputWidth;
    int gradOutStartPos = gid * outputHeight * outputWidth;

    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            int localPos = i * outputWidth + j;
            int gradIndex = (int) indices[gradOutStartPos + localPos];
            if (gradIndex != -1) {
                gradIn[gradInStartPos + gradIndex] += gradOut[gradOutStartPos + localPos];
            }
        }
    }
}

__kernel void avg_pool_update_out(__global float *input,
                                  __global float *output,
                                  const int inputHeight, const int inputWidth,
                                  const int outputHeight, const int outputWidth,
                                  const int kernelH, const int kernelW,
                                  const int strideH, const int strideW,
                                  const int paddingH, const int paddingW,
                                  const int includePadding)
{
    int gid = get_global_id(0); // [batch] * inputPlane
    int inputStartPos = gid * inputHeight * inputWidth;
    int outputStartPos = gid * outputHeight * outputWidth;
    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            int hStart = i * strideH - paddingH;
            int wStart = j * strideW - paddingW;
            int hEnd = min(hStart + kernelH, inputHeight + paddingH);
            int wEnd = min(wStart + kernelW, inputWidth + paddingW);

            int poolSize = (hEnd - hStart) * (wEnd - wStart);

            hStart = max(hStart, 0);
            wStart = max(wStart, 0);
            hEnd = min(hEnd, inputHeight);
            wEnd = min(wEnd, inputWidth);

            int itemsInPool = (includePadding == 0) ? (hEnd - hStart) * (wEnd - wStart) : poolSize;

            float sum = 0;

            for (int y = hStart; y < hEnd; y++) {
                for (int x = wStart; x < wEnd; x++) {
                    sum += input[inputStartPos + y * inputWidth + x];
                }
            }
            output[outputStartPos + i * outputWidth + j] = sum / ((float) itemsInPool);
        }
    }
}

__kernel void avg_pool_update_grad_in(__global float *gradIn,
                                      __global float *gradOut,
                                      const int inputHeight, const int inputWidth,
                                      const int outputHeight, const int outputWidth,
                                      const int kernelH, const int kernelW,
                                      const int strideH, const int strideW,
                                      const int paddingH, const int paddingW,
                                      const int includePadding)
{
    int gid = get_global_id(0); // [batch] * inputPlane
    int gradInStartPos = gid * inputHeight * inputWidth;
    int gradOutStartPos = gid * outputHeight * outputWidth;
    for (int i = 0; i < outputHeight; i++) {
        for (int j = 0; j < outputWidth; j++) {
            int hStart = i * strideH - paddingH;
            int wStart = j * strideW - paddingW;
            int hEnd = min(hStart + kernelH, inputHeight + paddingH);
            int wEnd = min(wStart + kernelW, inputWidth + paddingW);

            int poolSize = (hEnd - hStart) * (wEnd - wStart);

            hStart = max(hStart, 0);
            wStart = max(wStart, 0);
            hEnd = min(hEnd, inputHeight);
            wEnd = min(wEnd, inputWidth);

            int itemsInPool = (includePadding == 0) ? (hEnd - hStart) * (wEnd - wStart) : poolSize;

            int gradOutPos = gradOutStartPos + i * outputWidth + j;

            for (int y = hStart; y < hEnd; y++) {
                for (int x = wStart; x < wEnd; x++) {
                    gradIn[gradInStartPos + y * inputWidth + x] += gradOut[gradOutPos] / ((float) itemsInPool);
                }
            }
        }
    }
}

__kernel void partial_fast_sum(__global float *tensor,
                                __local float *temp,
                                const int length,
                                __global float *res)
{
    int gid = get_global_id(0);
    float acc = 0;
    while (gid < length) {
        float element = tensor[gid];
        acc += element;
        gid += get_global_size(0);
    }

    int lid = get_local_id(0);
    temp[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2) {
        if (lid < offset) {
            temp[lid] += temp[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        res[get_group_id(0)] = temp[0];
    }
}


__kernel void upsample_nearest_update_out(__global float *input,
                                          __global float *output,
                                          __global int *output_shape,
                                          const int scale
) {
    int gid = get_global_id(0);

    int stride0 = output_shape[3] * output_shape[2] * output_shape[1];
    int strideIn0 = (output_shape[3] / scale) * (output_shape[2] / scale) * output_shape[1];
    int stride1 = output_shape[3] * output_shape[2];
    int strideIn1 = (output_shape[3] / scale) * (output_shape[2] / scale);
    int stride2 = output_shape[3];
    int strideIn2 = output_shape[3] / scale;

    int iin0 = gid / stride0;
    int iin1 = ((gid / stride1) % output_shape[1]);
    int iin2 = ((gid / stride2) % output_shape[2]) / scale;
    int iin3 = gid % output_shape[3] / scale;

    int isrc = iin0 * strideIn0 + iin1 * strideIn1 + iin2 * strideIn2 + iin3;

    output[gid] = input[isrc];
}


__kernel void upsample_nearest_update_grad(__global float *gradIn,
                                           __global float *gradOut,
                                           __global int *gradInShape,
                                           const int scale
) {
    int gid = get_global_id(0);

    int stride0 = gradInShape[3] * gradInShape[2] * gradInShape[1];
    int strideOut0 = gradInShape[3] * scale * gradInShape[2] * scale * gradInShape[1];
    int stride1 = gradInShape[3] * gradInShape[2];
    int strideOut1 = gradInShape[3] * scale * gradInShape[2] * scale;
    int stride2 = gradInShape[3];
    int strideOut2 = gradInShape[3] * scale;

    int iin0 = gid / stride0;
    int iin1 = ((gid / stride1) % gradInShape[1]);
    int iin2 = ((gid / stride2) % gradInShape[2]);
    int iin3 = gid % gradInShape[3];

    for (int y = 0; y < scale; y++) {
        for (int x = 0; x < scale; x++) {
            int ioutx = iin2 * scale + x;
            int iouty = iin3 * scale + y;
            int isrc = iin0 * strideOut0 + iin1 * strideOut1 + ioutx * strideOut2 + iouty;
            gradIn[gid] += gradOut[isrc];
        }
    }
}
