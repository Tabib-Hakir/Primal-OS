const char* const VixlJniHelpersResults = {
  "       0: 2d e9 e0 4d   push.w {r5, r6, r7, r8, r10, r11, lr}\n"
  "       4: 2d ed 10 8a   vpush {s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31}\n"
  "       8: 81 b0         sub sp, #4\n"
  "       a: 00 90         str r0, [sp]\n"
  "       c: 19 91         str r1, [sp, #100]\n"
  "       e: 8d ed 1a 0a   vstr s0, [sp, #104]\n"
  "      12: 1b 92         str r2, [sp, #108]\n"
  "      14: 1c 93         str r3, [sp, #112]\n"
  "      16: 88 b0         sub sp, #32\n"
  "      18: ad f5 80 5d   sub.w sp, sp, #4096\n"
  "      1c: 08 98         ldr r0, [sp, #32]\n"
  "      1e: 1f 98         ldr r0, [sp, #124]\n"
  "      20: 21 98         ldr r0, [sp, #132]\n"
  "      22: ff 98         ldr r0, [sp, #1020]\n"
  "      24: dd f8 00 04   ldr.w r0, [sp, #1024]\n"
  "      28: dd f8 fc cf   ldr.w r12, [sp, #4092]\n"
  "      2c: 0d f5 80 5c   add.w r12, sp, #4096\n"
  "      30: dc f8 00 c0   ldr.w r12, [r12]\n"
  "      34: d9 f8 00 c2   ldr.w r12, [r9, #512]\n"
  "      38: dc f8 80 00   ldr.w r0, [r12, #128]\n"
  "      3c: 08 90         str r0, [sp, #32]\n"
  "      3e: 1f 90         str r0, [sp, #124]\n"
  "      40: 21 90         str r0, [sp, #132]\n"
  "      42: ff 90         str r0, [sp, #1020]\n"
  "      44: cd f8 00 04   str.w r0, [sp, #1024]\n"
  "      48: cd f8 fc cf   str.w r12, [sp, #4092]\n"
  "      4c: 4d f8 04 5d   str r5, [sp, #-4]!\n"
  "      50: 0d f5 80 55   add.w r5, sp, #4096\n"
  "      54: c5 f8 04 c0   str.w r12, [r5, #4]\n"
  "      58: 5d f8 04 5b   ldr r5, [sp], #4\n"
  "      5c: 4f f0 ff 0c   mov.w r12, #255\n"
  "      60: cd f8 30 c0   str.w r12, [sp, #48]\n"
  "      64: 6f f0 7f 4c   mvn r12, #4278190080\n"
  "      68: cd f8 30 c0   str.w r12, [sp, #48]\n"
  "      6c: cd f8 30 c0   str.w r12, [sp, #48]\n"
  "      70: cd f8 30 c0   str.w r12, [sp, #48]\n"
  "      74: 0c 90         str r0, [sp, #48]\n"
  "      76: dd f8 30 c0   ldr.w r12, [sp, #48]\n"
  "      7a: cd f8 34 c0   str.w r12, [sp, #52]\n"
  "      7e: 0d f5 80 5c   add.w r12, sp, #4096\n"
  "      82: c9 f8 00 c2   str.w r12, [r9, #512]\n"
  "      86: c9 f8 00 d2   str.w sp, [r9, #512]\n"
  "      8a: d0 f8 30 e0   ldr.w lr, [r0, #48]\n"
  "      8e: f0 47         blx lr\n"
  "      90: dd f8 2c c0   ldr.w r12, [sp, #44]\n"
  "      94: cd f8 30 c0   str.w r12, [sp, #48]\n"
  "      98: d9 f8 00 c2   ldr.w r12, [r9, #512]\n"
  "      9c: cd f8 2c c0   str.w r12, [sp, #44]\n"
  "      a0: dd f8 2c c0   ldr.w r12, [sp, #44]\n"
  "      a4: cd f8 30 c0   str.w r12, [sp, #48]\n"
  "      a8: 48 46         mov r0, r9\n"
  "      aa: cd f8 30 90   str.w r9, [sp, #48]\n"
  "      ae: 04 46         mov r4, r0\n"
  "      b0: 0d f1 30 0c   add.w r12, sp, #48\n"
  "      b4: bb f1 00 0f   cmp.w r11, #0\n"
  "      b8: 18 bf         it ne\n"
  "      ba: e3 46         movne r11, r12\n"
  "      bc: 0d f1 30 0b   add.w r11, sp, #48\n"
  "      c0: 5f ea 0b 00   movs.w r0, r11\n"
  "      c4: 18 bf         it ne\n"
  "      c6: 0c a8         addne r0, sp, #48\n"
  "      c8: dd f8 40 c0   ldr.w r12, [sp, #64]\n"
  "      cc: bc f1 00 0f   cmp.w r12, #0\n"
  "      d0: 18 bf         it ne\n"
  "      d2: 0d f1 40 0c   addne.w r12, sp, #64\n"
  "      d6: cd f8 30 c0   str.w r12, [sp, #48]\n"
  "      da: 5f ea 0b 00   movs.w r0, r11\n"
  "      de: 18 bf         it ne\n"
  "      e0: 00 a8         addne r0, sp, #0\n"
  "      e2: 0d f2 04 40   addw r0, sp, #1028\n"
  "      e6: bb f1 00 0f   cmp.w r11, #0\n"
  "      ea: 08 bf         it eq\n"
  "      ec: 58 46         moveq r0, r11\n"
  "      ee: 0d f2 04 4c   addw r12, sp, #1028\n"
  "      f2: bb f1 00 0f   cmp.w r11, #0\n"
  "      f6: 18 bf         it ne\n"
  "      f8: e3 46         movne r11, r12\n"
  "      fa: d9 f8 9c c0   ldr.w r12, [r9, #156]\n"
  "      fe: bc f1 00 0f   cmp.w r12, #0\n"
  "     102: 71 d1         bne 0x1e8     @ imm = #226\n"
  "     104: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     108: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     10c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     110: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     114: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     118: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     11c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     120: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     124: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     128: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     12c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     130: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     134: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     138: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     13c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     140: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     144: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     148: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     14c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     150: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     154: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     158: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     15c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     160: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     164: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     168: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     16c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     170: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     174: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     178: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     17c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     180: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     184: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     188: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     18c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     190: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     194: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     198: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     19c: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1a0: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1a4: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1a8: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1ac: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1b0: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1b4: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1b8: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1bc: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1c0: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1c4: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1c8: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1cc: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1d0: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1d4: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1d8: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1dc: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1e0: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1e4: 00 f0 02 b8   b.w 0x1ec     @ imm = #4\n"
  "     1e8: 00 f0 1b b8   b.w 0x222     @ imm = #54\n"
  "     1ec: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1f0: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1f4: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1f8: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     1fc: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     200: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     204: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     208: cd f8 ff c7   str.w r12, [sp, #2047]\n"
  "     20c: 0d f5 80 5d   add.w sp, sp, #4096\n"
  "     210: 08 b0         add sp, #32\n"
  "     212: 01 b0         add sp, #4\n"
  "     214: bd ec 10 8a   vpop {s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31}\n"
  "     218: bd e8 e0 4d   pop.w {r5, r6, r7, r8, r10, r11, lr}\n"
  "     21c: d9 f8 24 80   ldr.w r8, [r9, #36]\n"
  "     220: 70 47         bx lr\n"
  "     222: d9 f8 9c 00   ldr.w r0, [r9, #156]\n"
  "     226: d9 f8 d0 e2   ldr.w lr, [r9, #720]\n"
  "     22a: f0 47         blx lr\n"
};

const char* const VixlLoadFromOffsetResults = {
  "       0: e2 68         ldr r2, [r4, #12]\n"
  "       2: d4 f8 ff 2f   ldr.w r2, [r4, #4095]\n"
  "       6: 04 f5 80 52   add.w r2, r4, #4096\n"
  "       a: 12 68         ldr r2, [r2]\n"
  "       c: 04 f5 80 12   add.w r2, r4, #1048576\n"
  "      10: d2 f8 a4 20   ldr.w r2, [r2, #164]\n"
  "      14: 4f f4 80 52   mov.w r2, #4096\n"
  "      18: c0 f2 10 02   movt r2, #16\n"
  "      1c: 22 44         add r2, r4\n"
  "      1e: 12 68         ldr r2, [r2]\n"
  "      20: 4f f4 80 5c   mov.w r12, #4096\n"
  "      24: c0 f2 10 0c   movt r12, #16\n"
  "      28: 64 44         add r4, r12\n"
  "      2a: 24 68         ldr r4, [r4]\n"
  "      2c: a2 89         ldrh r2, [r4, #12]\n"
  "      2e: b4 f8 ff 2f   ldrh.w r2, [r4, #4095]\n"
  "      32: 04 f5 80 52   add.w r2, r4, #4096\n"
  "      36: 12 88         ldrh r2, [r2]\n"
  "      38: 04 f5 80 12   add.w r2, r4, #1048576\n"
  "      3c: b2 f8 a4 20   ldrh.w r2, [r2, #164]\n"
  "      40: 4f f4 80 52   mov.w r2, #4096\n"
  "      44: c0 f2 10 02   movt r2, #16\n"
  "      48: 22 44         add r2, r4\n"
  "      4a: 12 88         ldrh r2, [r2]\n"
  "      4c: 4f f4 80 5c   mov.w r12, #4096\n"
  "      50: c0 f2 10 0c   movt r12, #16\n"
  "      54: 64 44         add r4, r12\n"
  "      56: 24 88         ldrh r4, [r4]\n"
  "      58: d4 e9 03 23   ldrd r2, r3, [r4, #12]\n"
  "      5c: d4 e9 ff 23   ldrd r2, r3, [r4, #1020]\n"
  "      60: 04 f5 80 62   add.w r2, r4, #1024\n"
  "      64: d2 e9 00 23   ldrd r2, r3, [r2]\n"
  "      68: 04 f5 80 22   add.w r2, r4, #262144\n"
  "      6c: d2 e9 29 23   ldrd r2, r3, [r2, #164]\n"
  "      70: 4f f4 80 62   mov.w r2, #1024\n"
  "      74: c0 f2 04 02   movt r2, #4\n"
  "      78: 22 44         add r2, r4\n"
  "      7a: d2 e9 00 23   ldrd r2, r3, [r2]\n"
  "      7e: 4f f4 80 6c   mov.w r12, #1024\n"
  "      82: c0 f2 04 0c   movt r12, #4\n"
  "      86: 64 44         add r4, r12\n"
  "      88: d4 e9 00 45   ldrd r4, r5, [r4]\n"
  "      8c: dc f8 0c 00   ldr.w r0, [r12, #12]\n"
  "      90: a4 f5 80 12   sub.w r2, r4, #1048576\n"
  "      94: d2 f8 a4 20   ldr.w r2, [r2, #164]\n"
  "      98: 94 f9 0c 20   ldrsb.w r2, [r4, #12]\n"
  "      9c: 22 7b         ldrb r2, [r4, #12]\n"
  "      9e: b4 f9 0c 20   ldrsh.w r2, [r4, #12]\n"
};

const char* const VixlStoreToOffsetResults = {
  "       0: e2 60         str r2, [r4, #12]\n"
  "       2: c4 f8 ff 2f   str.w r2, [r4, #4095]\n"
  "       6: 04 f5 80 5c   add.w r12, r4, #4096\n"
  "       a: cc f8 00 20   str.w r2, [r12]\n"
  "       e: 04 f5 80 1c   add.w r12, r4, #1048576\n"
  "      12: cc f8 a4 20   str.w r2, [r12, #164]\n"
  "      16: 4f f4 80 5c   mov.w r12, #4096\n"
  "      1a: c0 f2 10 0c   movt r12, #16\n"
  "      1e: a4 44         add r12, r4\n"
  "      20: cc f8 00 20   str.w r2, [r12]\n"
  "      24: 4f f4 80 5c   mov.w r12, #4096\n"
  "      28: c0 f2 10 0c   movt r12, #16\n"
  "      2c: a4 44         add r12, r4\n"
  "      2e: cc f8 00 40   str.w r4, [r12]\n"
  "      32: a2 81         strh r2, [r4, #12]\n"
  "      34: a4 f8 ff 2f   strh.w r2, [r4, #4095]\n"
  "      38: 04 f5 80 5c   add.w r12, r4, #4096\n"
  "      3c: ac f8 00 20   strh.w r2, [r12]\n"
  "      40: 04 f5 80 1c   add.w r12, r4, #1048576\n"
  "      44: ac f8 a4 20   strh.w r2, [r12, #164]\n"
  "      48: 4f f4 80 5c   mov.w r12, #4096\n"
  "      4c: c0 f2 10 0c   movt r12, #16\n"
  "      50: a4 44         add r12, r4\n"
  "      52: ac f8 00 20   strh.w r2, [r12]\n"
  "      56: 4f f4 80 5c   mov.w r12, #4096\n"
  "      5a: c0 f2 10 0c   movt r12, #16\n"
  "      5e: a4 44         add r12, r4\n"
  "      60: ac f8 00 40   strh.w r4, [r12]\n"
  "      64: c4 e9 03 23   strd r2, r3, [r4, #12]\n"
  "      68: c4 e9 ff 23   strd r2, r3, [r4, #1020]\n"
  "      6c: 04 f5 80 6c   add.w r12, r4, #1024\n"
  "      70: cc e9 00 23   strd r2, r3, [r12]\n"
  "      74: 04 f5 80 2c   add.w r12, r4, #262144\n"
  "      78: cc e9 29 23   strd r2, r3, [r12, #164]\n"
  "      7c: 4f f4 80 6c   mov.w r12, #1024\n"
  "      80: c0 f2 04 0c   movt r12, #4\n"
  "      84: a4 44         add r12, r4\n"
  "      86: cc e9 00 23   strd r2, r3, [r12]\n"
  "      8a: 4f f4 80 6c   mov.w r12, #1024\n"
  "      8e: c0 f2 04 0c   movt r12, #4\n"
  "      92: a4 44         add r12, r4\n"
  "      94: cc e9 00 45   strd r4, r5, [r12]\n"
  "      98: cc f8 0c 00   str.w r0, [r12, #12]\n"
  "      9c: a4 f5 80 1c   sub.w r12, r4, #1048576\n"
  "      a0: cc f8 a4 20   str.w r2, [r12, #164]\n"
  "      a4: 22 73         strb r2, [r4, #12]\n"
};