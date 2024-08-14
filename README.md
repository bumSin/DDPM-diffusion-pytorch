# Denoising Diffusion Probabilistic Models (DDPM) Implementation from Scratch in Pytorch

Welcome to the DDPM Implementation repository! This repository contains a comprehensive implementation of Denoising Diffusion Probabilistic Models (DDPM) from scratch. This implementation is inspired by the Hugging Face’s DDPM implementation and aims to provide an in-depth understanding of the model’s architecture and functionality.

Table of Contents

	•	Introduction
	•	Background on DDPM
	•	Implementation Details

Introduction

Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn to generate high-quality samples from noise. They work by gradually denoising a sample through a series of steps, starting from a noisy image and progressively refining it to produce a clear output.

This repository contains a full implementation of DDPM, covering the core components, training procedures, and sampling algorithms. The goal is to provide a clear and educational example of how DDPM works under the hood.

![image](https://github.com/user-attachments/assets/f6b5e41b-48ff-4c33-92a0-cd97b94d77a3)

![image](https://github.com/user-attachments/assets/e8978f9e-e6b5-4fce-8e49-50d7bed96d78)

A few examples of what can be done usingh DDPMs only. Keep in mind that this is before all these fancy Generatrive models came around.

<img width="821" alt="image" src="https://github.com/user-attachments/assets/a96ccc33-cadf-4bbe-b655-df556994d6cd">

<img width="299" alt="image" src="https://github.com/user-attachments/assets/a07d9e42-27a6-4dcf-bf7f-ab3e83ea9bbd">

Background on DDPM

DDPMs are based on the concept of diffusion processes and probabilistic modeling. Here’s a detailed breakdown:

	1.	Diffusion Process: The forward process in DDPM gradually adds Gaussian noise to an image over a series of steps until the image is almost pure noise. The reverse process learns to recover the original image from this noisy version.
	2.	Denoising Objective: The core idea is to train a neural network to predict the noise added at each step of the forward diffusion process. This is achieved by minimizing the difference between the predicted noise and the actual noise added.
	3.	Sampling: Once trained, the model can generate new samples by starting from pure noise and applying the learned reverse diffusion process to produce a sample that resembles the data distribution.

Model Architecture

The DDPM architecture typically involves:

	•	Encoder: Extracts features from the input image.
	•	Noise Estimator: A neural network that predicts the noise added at each diffusion step.
	•	Decoder: Reconstructs the image from the noisy version.

Implementation Details

This implementation includes the following components:

	•	Data Preparation: Code for loading and preprocessing image data.
	•	Model Definition: Neural network architecture for noise estimation.
	•	Training Loop: Procedures for training the model on the dataset.
	•	Sampling Algorithm: Code to generate samples from the trained model.






## References
1. https://huggingface.co/blog/annotated-diffusion
