# Physics-Informed Neural Networks (PINNs) for Rainfall-Runoff Modeling

## 📦 Package Contents

This package contains comprehensive documentation and implementation of a Physics-Informed Neural Network for simulating rainfall-runoff on an inclined plane.

### Files Included

1. **PINNs_Rainfall_Runoff_DOCUMENTED.ipynb** - Main Jupyter notebook with complete implementation
2. **PINNs_Documentation.md** - Comprehensive 50+ page reference documentation
3. **README.md** - This file (quick start guide)

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
torch >= 1.10
numpy >= 1.20
matplotlib >= 3.4
tqdm
jupyter
```

### Installation

```bash
# Install dependencies
pip install torch numpy matplotlib tqdm jupyter

# Open notebook
jupyter notebook PINNs_Rainfall_Runoff_DOCUMENTED.ipynb
```

### Running the Simulation

1. Open the notebook in Jupyter
2. Run all cells (Cell → Run All)
3. Wait 15-20 minutes for training to complete
4. View the generated animations

---

## 📊 What This Model Does

The notebook implements a deep learning solution to the **2D Shallow Water Equations**, which govern the flow of water over terrain. Specifically, it simulates:

- **Rainfall** falling on an inclined plane (50 mm/hr for 3 minutes)
- **Runoff generation** as water accumulates and flows downslope  
- **Surface flow** with friction effects
- **Water depth and velocity** evolution over 5 minutes

### Key Outputs

- **3D Animation**: Water surface elevation over terrain
- **2D Animation**: Water depth contours with velocity vectors
- **Training Loss Plot**: Convergence history
- **Two MP4 video files**: Saved animations

---

## 🎯 Documentation Structure

### For Quick Users (30 min)

Just run the notebook! It's self-contained with:
- Clear code comments
- Section headers
- Progress indicators
- Automated visualization

### For Learning Users (2-3 hours)

Read **Sections 1-3** of `PINNs_Documentation.md`:
1. Overview - What PINNs are and why they're useful
2. Physical Model - The shallow water equations explained
3. Mathematical Formulation - How the neural network is trained

### For Advanced Users (Full Day)

Read the **complete documentation** (`PINNs_Documentation.md`):
- Sections 4-5: Implementation details and usage guide
- Section 6: Parameter tuning strategies
- Section 7: Troubleshooting common issues
- Section 8: Extensions and modifications

---

## 🔧 Key Parameters You Can Adjust

Located in the "Main Execution" section of the notebook:

### Rainfall Parameters
```python
rain_mm_hr = 50.0      # Intensity (mm/hr) - try 10, 50, 100
t_rain_end = 180.0     # Duration (seconds) - try 60, 180, 300
```

### Terrain Parameters  
```python
slope = 0.05           # Slope (fraction) - try 0.02, 0.05, 0.10
```

### Physics Parameters
```python
friction_factor = 0.01 # Friction coefficient - try 0.001, 0.01, 0.05
```

### Training Parameters
```python
n_epochs = 10000       # Training iterations - try 5000, 10000, 20000
learning_rate = 5e-4   # Learning rate - try 1e-4, 5e-4, 1e-3
```

### Loss Weights
```python
w_pde = 1.0           # PDE loss weight
w_ic = 150.0          # Initial condition weight
w_bc = 150.0          # Boundary condition weight
w_phys = 20.0         # Physical constraint weight
```

---

## 📈 Expected Results

### Training Progress

```
Epoch 0:     Loss ~250    (High initially)
Epoch 1000:  Loss ~12     (Rapidly decreasing)
Epoch 5000:  Loss ~2      (Slowing down)
Epoch 10000: Loss ~0.8    (Converged)
```

### Physical Results

- **Max water depth**: ~0.01-0.03 m (depends on parameters)
- **Max velocity**: ~0.1-0.5 m/s (downslope direction)
- **Flow pattern**: Uniform sheet flow down the incline
- **Drainage**: Water drains after rainfall stops

---

## 🔍 Understanding the Model

### The Physics-Informed Approach

Traditional simulation:
1. Discretize domain into grid cells
2. Solve PDEs numerically (finite difference/volume)
3. Step forward in time

**PINN approach**:
1. Neural network learns the solution directly
2. Trained to satisfy PDEs everywhere (via automatic differentiation)
3. Continuous representation (no grid!)

### Advantages of PINNs

✅ **Mesh-free**: No need to discretize the domain  
✅ **Differentiable**: Get velocities, accelerations easily  
✅ **Flexible**: Easy to change boundary conditions  
✅ **Generalizable**: Can interpolate between training scenarios  

### Limitations of PINNs

⚠️ **Training time**: 10-20 minutes vs. seconds for traditional methods  
⚠️ **Hyperparameter sensitivity**: Requires tuning loss weights  
⚠️ **Black box**: Harder to debug than explicit schemes  
⚠️ **Limited to smooth solutions**: Struggles with shocks/discontinuities  

---

## 📚 Learning Path

### Beginner (No ML Background)

1. **Run the notebook** first - see what it does
2. **Read Documentation Sections 1-2** - understand the physics
3. **Modify rainfall intensity** - observe effects
4. **Try different slopes** - build intuition

### Intermediate (Some ML Background)

1. **Read Documentation Sections 1-4** - full implementation details
2. **Experiment with loss weights** - understand training dynamics
3. **Modify the terrain function** - create valleys, ridges
4. **Add custom visualizations** - plot depth vs. time at a point

### Advanced (ML + PDEs)

1. **Read full documentation** - all sections
2. **Implement full Manning friction** - replace linear friction
3. **Add infiltration** - extend the physics
4. **Try transfer learning** - train on simple, fine-tune on complex
5. **Compare with traditional solver** - validate accuracy

---

## 🐛 Troubleshooting

### Problem: NaN Loss

**Cause**: Learning rate too high  
**Solution**:
```python
learning_rate = 1e-4  # Reduce from 5e-4
```

### Problem: Negative Water Depths

**Cause**: Physical constraint weight too low  
**Solution**:
```python
w_phys = 50.0  # Increase from 20.0
```

### Problem: Water Leaks Through Walls

**Cause**: Boundary condition weight too low  
**Solution**:
```python
w_bc = 300.0   # Increase from 150.0
N_bc = 4000    # More boundary points
```

### Problem: Very Slow Training

**Cause**: Too many collocation points  
**Solution**:
```python
N_collocation = 5000   # Reduce from 10000
hid_dim = 32           # Smaller network
```

### Problem: Poor Convergence

**Cause**: Network too small or learning rate too low  
**Solution**:
```python
hid_dim = 128           # Increase from 64
learning_rate = 1e-3    # Increase from 5e-4
n_epochs = 20000        # More training
```

---

## 🎓 Concepts Explained

### What is a PINN?

A **Physics-Informed Neural Network** is a neural network that:
1. Takes coordinates (x, y, t) as input
2. Outputs physical quantities (depth, velocity)
3. Is trained to satisfy governing PDEs (via loss function)

Think of it as a "smart interpolator" that respects physics laws.

### How Does Training Work?

```
1. Sample random points in space-time
2. Neural network predicts (ζ, u, v) at those points
3. Use automatic differentiation to compute PDE residuals
4. Minimize residual (= how much PDE is violated)
5. Also minimize IC, BC, and physical constraint violations
6. Repeat for many iterations
```

### Why Multiple Loss Terms?

- **PDE loss**: Solution satisfies physics
- **IC loss**: Solution matches initial state
- **BC loss**: Solution respects boundaries
- **Physical loss**: Solution is physically realistic (h ≥ 0)

All must be satisfied simultaneously!

### What is Automatic Differentiation?

Traditional: Approximate derivatives with finite differences
```python
df_dx ≈ (f(x+h) - f(x)) / h  # Error ~O(h)
```

Automatic differentiation: Exact derivatives via chain rule
```python
df_dx = torch.autograd.grad(f, x, ...)[0]  # Exact!
```

This is crucial for PINNs - we need accurate PDE residuals.

---

## 🔬 Experiments to Try

### Easy Experiments (30 min each)

1. **Double the rainfall**: Set `rain_mm_hr = 100.0`
   - Observe: More runoff, deeper water, faster velocities

2. **Halve the friction**: Set `friction_factor = 0.005`
   - Observe: Water flows faster, less ponding

3. **Steeper slope**: Set `slope = 0.10` in terrain function
   - Observe: More channelized flow, faster drainage

### Medium Experiments (1-2 hours each)

4. **Shorter rainfall**: Set `t_rain_end = 60.0`
   - Compare: Total runoff volume, peak depth

5. **Different initial conditions**: Start with `h0_np = 0.001` (1mm)
   - Observe: How IC affects early-time behavior

6. **Valley terrain**: Modify terrain to create a central valley
   ```python
   z = 0.1 - 0.05*X - 0.1*(Y-0.5)**2
   ```
   - Observe: Water channels into valley

### Advanced Experiments (Half day each)

7. **Spatially-varying rainfall**: Rain only in a circular region
   ```python
   R = R_const if (Xc-0.5)**2 + (Yc-0.5)**2 < 0.1 else 0
   ```

8. **Time-varying rainfall**: Sinusoidal intensity
   ```python
   R = R_const * (1 + 0.5*sin(2*pi*Tc/60))
   ```

9. **Compare with analytical solution**: For very simple cases
   - Steady uniform flow on infinite plane has analytical solution
   - Compare PINN vs. analytical at late times

---

## 📖 Recommended Reading Order

### Day 1: Getting Started
- This README (you're here!)
- Run the notebook
- Documentation Sections 1-2

### Day 2: Understanding
- Documentation Section 3 (Mathematical Formulation)
- Documentation Section 4 (Implementation Details)
- Try modifying simple parameters

### Day 3: Mastery
- Documentation Sections 5-6 (Usage & Tuning)
- Implement one medium experiment
- Read Section 8 (Extensions)

### Day 4: Advanced
- Documentation Section 7 (Troubleshooting)
- Implement one advanced experiment
- Compare with literature

---

## 🎯 Success Criteria

You've mastered this notebook when you can:

✅ Explain what the shallow water equations represent  
✅ Describe how PINNs differ from traditional solvers  
✅ Identify which loss component is problematic from training logs  
✅ Adjust parameters to achieve desired behavior  
✅ Modify the terrain function for custom topography  
✅ Add a new physics term (infiltration, sediment, etc.)  
✅ Debug common training issues (NaN, leaks, negative depths)  
✅ Explain the trade-offs between PINN and traditional methods  

---

## 🤝 Getting Help

### If Something Doesn't Work

1. **Check the Troubleshooting section** (above and in full docs)
2. **Read error messages carefully** - they usually indicate the issue
3. **Start with default parameters** - make sure baseline works
4. **Change one thing at a time** - easier to identify problems

### If You Want to Learn More

1. **Read the full documentation** - `PINNs_Documentation.md`
2. **Consult the references** - papers on PINNs and shallow water
3. **Explore PINN repositories** - DeepXDE, NeuralPDE.jl, etc.
4. **Compare with traditional solvers** - understand trade-offs

---

## 📝 Citation

If you use this notebook in your research, please cite:

```
@software{pinn_rainfall_runoff_2026,
  title = {Physics-Informed Neural Networks for Rainfall-Runoff Modeling},
  author = {Documentation Team},
  year = {2026},
  note = {Jupyter notebook with comprehensive documentation}
}
```

---

## 📄 License

This notebook and documentation are provided for educational purposes.

---

## 🙏 Acknowledgments

- **PyTorch Team**: For the automatic differentiation framework
- **PINN Community**: For developing these methods
- **Hydrology Community**: For the physics understanding

---

## 📧 Feedback

If you find errors, have suggestions, or want to share your experiments, please provide feedback!

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Complete and tested

---

## Quick Reference Card

### File Sizes
- Notebook: ~30 MB
- Documentation: ~150 KB
- Generated MP4s: ~10 MB each

### Time Requirements
- First run: 20-25 min (training + visualization)
- Subsequent runs: 15-20 min
- Reading full docs: 3-4 hours
- Implementing extensions: 1-8 hours

### Hardware Requirements
- **Minimum**: CPU, 8GB RAM
- **Recommended**: GPU (CUDA), 16GB RAM
- **Optimal**: Modern GPU (T4, V100), 32GB RAM

### Key Shortcuts
- Run cell: `Shift + Enter`
- Run all: `Cell → Run All`
- Interrupt: `Kernel → Interrupt`
- Restart: `Kernel → Restart`

---

*Happy learning and experimenting with PINNs!* 🎉
