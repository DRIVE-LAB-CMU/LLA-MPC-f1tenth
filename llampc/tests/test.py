from llampc.rollout import DynamicBank
from llampc.rollout import ModelBank
from llampc.utils import Spline, Spline2D, Projection, Track
from llampc.planner import get_reference_trajectory_segment

def main(args=None):
    t = Track("sim_track.npz")
    print(t)
    x_ref, projidx = get_reference_trajectory_segment([0.0, 0.0], 0.5, t, 20, .2, 0)
    print(x_ref)




if __name__ == '__main__':
    main()