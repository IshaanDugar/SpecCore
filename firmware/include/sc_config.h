/* GENERATED from speccore.toml by `python tools/sc.py gen`. Edit the toml, not this file. */
#ifndef SC_CONFIG_H
#define SC_CONFIG_H

#define SC_VOCAB 2048
#define SC_EOT_ID 0
#define SC_DRAFT_DIM 128
#define SC_DRAFT_LAYERS 2
#define SC_DRAFT_HEADS 4
#define SC_DRAFT_HEAD_DIM 32
#define SC_DRAFT_HIDDEN 512
#define SC_DRAFT_CTX 128
#define SC_TARGET_DIM 256
#define SC_TARGET_LAYERS 6
#define SC_TARGET_HEADS 8
#define SC_TARGET_HEAD_DIM 32
#define SC_TARGET_HIDDEN 1024
#define SC_TARGET_CTX 128
#define SC_GAMMA 4
#define SC_K 5
#define SC_PRED_FIXED 0
#define SC_PRED_BIMODAL 1
#define SC_PRED_GSHARE 2
#define SC_PRED_PERCEPTRON 3
#define SC_PRED_TAGE 4
#define SC_PREDICTOR SC_PRED_FIXED
#define SC_MMIO_CYCLES 0xFFFF00A0u
#define SC_CLOCK_MHZ 100
#define SC_KV_V_TRANSPOSED 1

#define SC_LANES 8
#define SC_WEIGHT_PORT_BITS 32
#define SC_MUL_DSP 1
#define SC_OPC_CUSTOM0 0x0B
#define SC_F3_SETK 0
#define SC_F3_SETP 1
#define SC_F3_ACCRD 3
#define SC_F3_VDOTK 4
#define SC_F3_WPOP 5
#define SC_ISA_VDOTK 1
#define SC_ISA_SETSRC 0
#define SC_ISA_ACCQ 0
#define SC_ISA_VRESOLVE 0
#define SC_ISA_VDOTK4 0

#endif
