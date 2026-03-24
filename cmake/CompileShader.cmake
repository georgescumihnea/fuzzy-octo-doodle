if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE OR NOT DEFINED COMPILER OR NOT DEFINED MODE)
    message(FATAL_ERROR "CompileShader.cmake requires INPUT_FILE, OUTPUT_FILE, COMPILER, and MODE.")
endif()

if(MODE STREQUAL "glslc")
    execute_process(
        COMMAND "${COMPILER}" "${INPUT_FILE}" -o "${OUTPUT_FILE}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
elseif(MODE STREQUAL "glslangValidator")
    execute_process(
        COMMAND "${COMPILER}" -V "${INPUT_FILE}" -o "${OUTPUT_FILE}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
else()
    message(FATAL_ERROR "Unsupported shader compiler mode: ${MODE}")
endif()

if(NOT result EQUAL 0)
    message(FATAL_ERROR "Failed to compile shader ${INPUT_FILE}\n${output}\n${error}")
endif()
