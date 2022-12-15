import os
import numpy as np


class repoUtil:
    @classmethod
    def output_csv(cls, outdpath, data, name):
        output = os.path.join(outdpath, f"irtdata_{name}.csv")
        np.savetxt(output, data, delimiter=",", fmt="%.5f")
