# GNS for uncertainty quantification

## Scenario

Scenarios: we make a lot of MPM inputs for random field slope simulations. 
These are subset of entire simulation sets $ S_{MPM} \in S $. GNS simulates the entire samples. 

* Since you need to run GNS simulations to all samples, GNS needs to read the inputfiles and make `X0`. 
* For the selected samples, run MPM. We don't need preprocessing since the input files already follow the MPM input format.
* In general, we need to track the rollout results for both GNS and MPM to make probability distributions. For example, `trajectory1-gns.npz`, `trajectory1-mpm.npz`. 
* For the samples that runs both MPM and GNS, we may select a few of those samples and compare the rollout. 

## Design
**GNS:**
* Need to make X0. Here, determining initial 5 velocities are important design consideration. There can be two options.
    * Run MPM for 6 steps and make `trajectory.npz`. &rarr; might entail additional computation for initial MPM runs, but robust approach.
    > Note that you need to have geostatic stress inputs before this running 6 steps.
    * Select a MPM simulation result and use initial 5 velocities for the GNS input. &rarr; higly likely to produce inconsistent GNS results. Depending on material properties, entire mass might be static. When you impose a few initial velocities to this case, not sure what will happen. 
* Whatever option you choose from above, make `trajectory${i}-gns.npz` for GNS to read to make `X0`. 
* Read `trajectory.npz` and rollout. I alreay have the code for this which I used in inverse-multilayer project.

**MPM:**
* Run selected samples. 
* Save the results as `trajectory${i}-mpm.npz`
    * This might include a simulation with only 6 steps.
