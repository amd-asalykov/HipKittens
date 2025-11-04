# Run the following commands to profile the performance counters of kernels called within a Python file called ```test_python.py```.

rocprofv3 --pmc SQC_ICACHE_MISSES,SQC_DCACHE_REQ,SQC_DCACHE_HITS,SQC_DCACHE_MISSES,SQ_ACTIVE_INST_ANY,SQ_ACTIVE_INST_VMEM,SQ_WAIT_INST_ANY --output-format csv --output-file profiles_1 -d out -- python3 test_python.py 

rocprofv3 --pmc SQC_ICACHE_REQ,SQC_ICACHE_HITS,SQ_INSTS,SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_SMEM,SQ_INSTS_VMEM --output-format csv --output-file profiles_2 -d out -- python3 test_python.py 

rocprofv3 --pmc SPI_CSN_WAVE,SQ_WAVES,SQ_CYCLES,SQ_BUSY_CYCLES --output-format csv --output-file profiles_4 -d out -- python3 test_python.py 

rocprofv3 --pmc SPI_CSN_WAVE,SQ_WAVES,SQ_CYCLES,SQ_BUSY_CYCLES,SPI_CSN_BUSY,SPI_CSN_NUM_THREADGROUPS,SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS,TCC_REQ_sum,TCC_HIT_sum,TCC_MISS_sum,TCC_ATOMIC_sum,TCP_TOTAL_ACCESSES_sum,TCP_TOTAL_READ_sum,TCP_TOTAL_WRITE_sum --output-format csv --output-file profiles_3 -d out -- python3 test_python.py


