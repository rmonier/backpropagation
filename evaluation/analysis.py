import subprocess

def main():
    activations = ["sigmoid"]
    learning_rates = [0.01, 0.05, 0.1, 0.15, 0.2]
    momentums = [0, 0.1, 0.5, 0.7, 0.9]
    epochs = [10, 100, 200, 300, 400, 500, 1000]
    hidden_layers = [[9,5], [4,6], [9,5,6], [4,6,8], [2,3]]
    for activation in activations:
        for learning_rate in learning_rates:
            for momentum in momentums:
                for epoch in epochs:
                    for hidden_layer in hidden_layers:
                        hl = "[4; "
                        for i in hidden_layer:
                            hl += f"{i}; "
                        hl += "1]"
                        print(f"julia --project=. network/bp.jl \"datasets/split/A1-turbine\" \"{hl}\" {activation} {learning_rate} {momentum} {epoch}\n")
                        subprocess.run(f"julia --project=. network/bp.jl \"datasets/split/A1-turbine\" \"{hl}\" {activation} {learning_rate} {momentum} {epoch}")
                        print("\n")

                        hl = "[9; "
                        for i in hidden_layer:
                            hl += f"{i}; "
                        hl += "1]"
                        print(f"julia --project=. network/bp.jl \"datasets/split/A1-synthetic\" \"{hl}\" {activation} {learning_rate} {momentum} {epoch}\n")
                        subprocess.run(f"julia --project=. network/bp.jl \"datasets/split/A1-synthetic\" \"{hl}\" {activation} {learning_rate} {momentum} {epoch}")
                        print("\n")
                        
                        hl = "[9; "
                        for i in hidden_layer:
                            hl += f"{i}; "
                        hl += "1]"
                        print(f"julia --project=. network/bp.jl \"datasets/split/A1-top10s\" \"{hl}\" {activation} {learning_rate} {momentum} {epoch}\n")
                        subprocess.run(f"julia --project=. network/bp.jl \"datasets/split/A1-top10s\" \"{hl}\" {activation} {learning_rate} {momentum} {epoch}")
                        print("\n")

if __name__ == '__main__':
    main()
