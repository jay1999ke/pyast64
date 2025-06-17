
# pyx

a python3 compiler and asm optimizer

## Matrix multiplication tests

### Raw numbers (no optimization)
```
    version_flag = VERSION_CLASSIC

    P = 2048
    Q = 2048
    R = 2048
```
```

 Performance counter stats for './simple.exe':

       1,83,694.80 msec task-clock                #    0.999 CPUs utilized          
             6,099      context-switches          #    0.033 K/sec                  
                31      cpu-migrations            #    0.000 K/sec                  
            25,789      page-faults               #    0.140 K/sec                  
 7,12,56,93,97,546      cycles                    #    3.879 GHz                    
 7,84,89,04,69,999      instructions              #    1.10  insn per cycle         
   28,13,87,25,295      branches                  #  153.182 M/sec                  
         97,72,322      branch-misses             #    0.03% of all branches        

     183.941315620 seconds time elapsed

     183.553050000 seconds user
       0.139982000 seconds sys
```
--------------------------------
```
    version_flag = VERSION_IKJ

    P = 2048
    Q = 2048
    R = 2048
```
```
 Performance counter stats for './simple.exe':

         92,072.51 msec task-clock                #    1.000 CPUs utilized          
             1,154      context-switches          #    0.013 K/sec                  
                 4      cpu-migrations            #    0.000 K/sec                  
            24,618      page-faults               #    0.267 K/sec                  
 3,60,73,43,32,304      cycles                    #    3.918 GHz                    
 6,20,01,62,85,227      instructions              #    1.72  insn per cycle         
   26,06,33,70,075      branches                  #  283.074 M/sec                  
         69,66,799      branch-misses             #    0.03% of all branches        

      92.095818946 seconds time elapsed

      92.048590000 seconds user
       0.023998000 seconds sys
```
----
```
    version_flag = VERSION_BLOCKED

    P = 2048
    Q = 2048
    R = 2048
```
```
 Performance counter stats for './simple.exe':

         92,268.20 msec task-clock                #    1.000 CPUs utilized          
               352      context-switches          #    0.004 K/sec                  
                 1      cpu-migrations            #    0.000 K/sec                  
            24,620      page-faults               #    0.267 K/sec                  
 3,66,92,65,69,032      cycles                    #    3.977 GHz                    
 5,18,44,95,88,537      instructions              #    1.41  insn per cycle         
   27,45,65,96,278      branches                  #  297.574 M/sec                  
      29,62,01,656      branch-misses             #    1.08% of all branches        

      92.272298674 seconds time elapsed

      92.244466000 seconds user
       0.023998000 seconds sys

```
### Peephole #1

```
    version_flag = VERSION_IKJ

    P = 2048
    Q = 2048
    R = 2048
```
```

 Performance counter stats for './simple.exe':

         31,039.95 msec task-clock                #    0.999 CPUs utilized          
             1,125      context-switches          #    0.036 K/sec                  
                 4      cpu-migrations            #    0.000 K/sec                  
            24,622      page-faults               #    0.793 K/sec                  
 1,22,09,00,11,712      cycles                    #    3.933 GHz                    
 3,61,53,88,06,931      instructions              #    2.96  insn per cycle         
   25,92,67,28,954      branches                  #  835.270 M/sec                  
         87,61,764      branch-misses             #    0.03% of all branches        

      31.075674004 seconds time elapsed

      30.976127000 seconds user
       0.063967000 seconds sys

```
----

-------------------
-------------------
inspired by pyast64:
```
pyast64 is a Python 3 program that compiles a 
subset of the Python AST to x64-64 assembler. 
It's extremely restricted (read "a toy") 
but it's a nice proof of concept in any case. 
(http://benhoyt.com/writings/pyast64/)
