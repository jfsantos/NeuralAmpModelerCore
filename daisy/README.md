# Benchmarking tool for Electrosmith Daisy board

This folder contains a benchmarking tool for the Electrosmith Daisy board. To build it, clone this repo or create a submodule inside the [DaisyExamples](https://github.com/electro-smith/DaisyExamples) repo like so:

```
git clone https://github.com/electro-smith/DaisyExamples
cd DaisyExamples
git submodule update --init
git submodule add -b daisy https://github.com/jfsantos/NeuralAmpModelerCore seed/
```

To build, you need to have the Daisy toolchain installed and in your path, then just run `make` in the `NeuralAmpModelerCore/daisy` repo:

```
cd seed/NeuralAmpModelerCore/daisy
make
make program-dfu # to copy the code to the Daisy board via USB
```

The benchmarking code will read `.nam` files from the SD card (connected via a breakout board or the one in the Daisy Pod board) and run them one by one, reporting the results on the serial port. There are no checks to see if a given model will fit on the board, so the code might just hang if a model is too large.
