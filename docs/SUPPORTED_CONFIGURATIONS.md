# Configuration and tested-support matrix

| Capability | Tested support / restriction |
| --- | --- |
| CPU | DNANM/DNACT, double precision, unit protein masses, targeted tests |
| CUDA | GH200, mixed precision; DNANM and selected DNACT **global** bending/torsion stiffness, valid parameter files |
| CUDA angular model | Selected global nonzero cases tested; **per-bond** angular stiffness not qualified |
| CUDA list/sorting | `use_edge=1`, `CUDA_sort_every=0` in the identified campaign |
| Protein mass | Unit mass in native model units; non-unit mass not qualified |
| DNA feature flags | Unsupported disabled terms rejected, not silently ignored |
| CUDA `force_and_torque` observable | Unsupported request rejected explicitly; native CUDA force buffer export is not implemented |
| Pressure coupling, MC/NPT, arbitrary external forces | Outside the qualified release scope |
| Long-time equilibrium, production NVT | Not established by targeted tests or 4096-step diagnostics |

A rejected configuration is an intentional fail-closed safety property, **not** a sign that the feature is implemented. Do not infer generic CUDA, NPT or sorting support from a successful build.
