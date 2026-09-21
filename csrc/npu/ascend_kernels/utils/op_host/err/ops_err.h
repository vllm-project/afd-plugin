/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Parameter-validation log helpers for GatherSelectionSparseFlashAttention.
 * Thin wrappers over CAM OPS_LOG_E (no function-like macros).
 */
#pragma once

#ifndef OPS_UTILS_LOG_SUB_MOD_NAME
#define OPS_UTILS_LOG_SUB_MOD_NAME "GATHER_SELECTION_SPARSE_FLASH_ATTENTION"
#endif
#ifndef OPS_UTILS_LOG_PACKAGE_TYPE
#define OPS_UTILS_LOG_PACKAGE_TYPE "[CAM]"
#endif

#include <string>
#include "../ops_error.h"
#include "../ops_log.h"
#include "log/log.h"

#define OPS_INNER_ERR_STUB(ERR_CODE_STR, OPS_DESC, FMT, ...)                                                           \
    do {                                                                                                               \
        OpLogSub(OP, DLOG_ERROR, OPS_DESC, FMT, ##__VA_ARGS__);                                                    \
        REPORT_INNER_ERR_MSG(ERR_CODE_STR, FMT, ##__VA_ARGS__);                                                        \
    } while (0)


#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E89999", OPS_DESC, __VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E69999", OPS_DESC, __VA_ARGS__)

inline void OpLogeForInvalidShape(const std::string &opName, const std::string &paramName,
                                  const std::string &incorrectShape, const std::string &correctShape)
{
    OPS_LOG_E(opName.c_str(), "Parameter %s has incorrect shape %s. It should be %s.", paramName.c_str(),
              incorrectShape.c_str(), correctShape.c_str());
}

inline void OpLogeForInvalidShapeWithReason(const std::string &opName, const std::string &paramName,
                                            const std::string &incorrectShape, const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameter %s has incorrect shape %s. Reason: %s.", paramName.c_str(),
              incorrectShape.c_str(), reason.c_str());
}

inline void OpLogeForInvalidShapesWithReason(const std::string &opName, const std::string &paramNames,
                                             const std::string &incorrectShapes, const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameters %s have incorrect shapes %s. Reason: %s.", paramNames.c_str(),
              incorrectShapes.c_str(), reason.c_str());
}

inline void OpLogeForInvalidShapedimWithReason(const std::string &opName, const std::string &paramName,
                                               const std::string &incorrectDim, const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameter %s has incorrect dim %s. Reason: %s.", paramName.c_str(),
              incorrectDim.c_str(), reason.c_str());
}

inline void OpLogeForInvalidShapesizeWithReason(const std::string &opName, const std::string &paramName,
                                                const std::string &incorrectSize, const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameter %s has incorrect shape size %s. Reason: %s.", paramName.c_str(),
              incorrectSize.c_str(), reason.c_str());
}

inline void OpLogeForInvalidDtypeWithReason(const std::string &opName, const std::string &paramName,
                                            const std::string &incorrectDtype, const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameter %s has incorrect dtype %s. Reason: %s.", paramName.c_str(),
              incorrectDtype.c_str(), reason.c_str());
}

inline void OpLogeForInvalidDtypesWithReason(const std::string &opName, const std::string &paramNames,
                                             const std::string &incorrectDtypes, const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameters %s have incorrect dtypes %s. Reason: %s.", paramNames.c_str(),
              incorrectDtypes.c_str(), reason.c_str());
}

inline void OpLogeForInvalidValue(const std::string &opName, const std::string &paramName,
                                  const std::string &incorrectValue, const std::string &correctValue)
{
    OPS_LOG_E(opName.c_str(), "Parameter %s has incorrect value %s. It should be %s.", paramName.c_str(),
              incorrectValue.c_str(), correctValue.c_str());
}

inline void OpLogeForInvalidValueWithReason(const std::string &opName, const std::string &paramName,
                                            const std::string &incorrectValue, const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameter %s has incorrect value %s. Reason: %s.", paramName.c_str(),
              incorrectValue.c_str(), reason.c_str());
}

inline void OpLogeForInvalidValuesWithReason(const std::string &opName, const std::string &paramNames,
                                             const std::string &incorrectValues, const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameters %s have incorrect values %s. Reason: %s.", paramNames.c_str(),
              incorrectValues.c_str(), reason.c_str());
}

inline void OpLogeForInvalidArgumentWithReason(const std::string &opName, const std::string &paramName,
                                               const std::string &reason)
{
    OPS_LOG_E(opName.c_str(), "Parameter %s is invalid. Reason: %s.", paramName.c_str(), reason.c_str());
}
