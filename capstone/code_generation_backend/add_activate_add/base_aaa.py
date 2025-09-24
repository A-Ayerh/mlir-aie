import sys
import numpy as np
from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.device.tile import AnyComputeTile
import aie.iron as iron


@iron.jit(is_placed=False)
def base_aaa(input0, output):

    # Object fifos goes here... --------------------------------------------\/
    
    # Core functions go here... --------------------------------------------\/

    # Workers go here... ---------------------------------------------------\/

    # Runtime data movement goes here... -----------------------------------\/
    
    # Create and return Program --------------------------------------------\/
    # ... = Program(iron.get_current_device())
    # ... .resolve_program(SequentialPlacer())
    # return ...

    print()


def main():
    base_aaa()

if __name__ == "__main__":
    main()