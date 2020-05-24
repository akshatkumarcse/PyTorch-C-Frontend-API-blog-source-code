#include<iostream>
#include<torch/torch.h>


// build a neural network similar to how you would do it with Pytorch 

struct Model : torch::nn::Module {      //Inheriting from the torch::nn::Module class.

    // Constructor
    Model() {
        // construct and register your layers
        input = register_module("input",torch::nn::Linear(8,64));
        hidden = register_module("hidden",torch::nn::Linear(64,64));
        output = register_module("output",torch::nn::Linear(64,1));
    }

    // the forward operation (how data will flow from layer to layer)
    torch::Tensor forward(torch::Tensor X){
        // let's pass relu 
        X = torch::relu(input->forward(X));
        X = torch::relu(hidden->forward(X));
        X = torch::sigmoid(output->forward(X));
        
        // return the output
        return X;
    }

    torch::nn::Linear input{nullptr},hidden{nullptr},output{nullptr};



};


int main(){

    Model model;
    
    auto input = torch::rand({8,});

    auto output = model.forward(input);

    std::cout << input << std::endl;
    std::cout << output << std::endl;

    std::cin.get();
}