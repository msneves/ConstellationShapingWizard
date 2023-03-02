
# Constellation Shaping Wizard

This repository deploys a webapp where one can visually analyze the process of end-to-end learning of the probabilities, constellation geometry, and demapper of a communication system. 

The user can change main system parameters such as: 
- the operating signal-to-noise ratio;
- the constellation order;
- whether to learn both probabilities and geometry, only one of those alone, or neither;
- choosing a NN demapper versus a maximum a posteriori (MAP) demapper.

The project only needs to be running in a host machine and can thus be assessed through browser by any machine on the network.
## Author

[@msneves](https://www.github.com/msneves)

Manuel S. Neves, 
PhD Candidate at University of Aveiro, Scientific Researcher at Instituto de Telecomunicações, Aveiro, Portugal


## Installation

In the host machine/server, run the following command to install the required packages:
```bash
  pip install numpy tensorflow pandas matplotlib streamlit
```

After, just change your current working directory to: 

```bash
  <path to local repository>\AE_GUI\
```

And then run:
```bash
  streamlit run main.py
```

And you're done! On the console, the address to access the webapp should be presented.

## Support

For support, questions and suggestions email msneves@ua.pt

