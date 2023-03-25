# RateTunableCBFs

## Simulation Results

## Leader follower
The follower's objective is to keep leader inside the field of view(FoV) while maximizing the reward which maximum when leader is at the center of FoV. The leader moves independently and therefore FoV is a time-varying constraint for the follower. Our proposed method is able to adapt the parameters for better performance. 

When no input bounds are imposed, the proposed method is better able to maintain the leader at center of FoV. When input bounds are present, the default CBF parameters fail to ensure a solution of QP controller soon after the simulation starts. The proposed method on the other hand is able to quickly adapt parameters to maintain FoV constraint while still satisfying input bounds. Due to limited actuation, the algorithm learns that it is not feasible to focus on performance and moves in almost a straight line which ensures that leader remains inside FoV.

|  | No Adaptation | With adaptation |
| --------------| -------------------| -----------------|
| No input bound | ![no_bound_no_adapt](https://user-images.githubusercontent.com/19849515/227720687-a3f1142f-7004-4c7d-b572-b90d29fb6d71.gif) | ![adapt_no_bound](https://user-images.githubusercontent.com/19849515/227720750-57c96b16-f799-44ae-b97d-8fe5f21349dc.gif) |
| Input Bound | ![no_adapt_with_bound](https://user-images.githubusercontent.com/19849515/227720777-b5909dec-079d-4dc3-8562-d94a4118f344.gif) | ![adapt_with_bound](https://user-images.githubusercontent.com/19849515/227720799-cfb15944-8b8e-425c-82a5-33349336f1b3.gif) |


## Trust based Adaptation
### Scenario 1:
| Constant CBF parameter | Trust based adaptation |
| --------------| -------------------|
| ![no_adapt](https://user-images.githubusercontent.com/19849515/227721767-75d395db-ca03-47b3-a1cd-08284ad61e6d.gif)| ![trust_adapt](https://user-images.githubusercontent.com/19849515/227721791-3695f2fa-b748-4309-92fa-3d8fb2dfe6f3.gif) |

### Scenario 2: 
| Constant CBF parameter | Trust based adaptation |
| --------------| -------------------|
| ![trust_fixed_parameter](https://user-images.githubusercontent.com/19849515/227721066-e2492b6c-eb11-4a0a-86df-677381d555c3.gif) | ![trust_adaptive](https://user-images.githubusercontent.com/19849515/227721079-a36a6ab0-cb4d-4f57-84d8-bf4628020085.gif) |


