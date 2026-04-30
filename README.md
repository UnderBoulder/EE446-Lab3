# EE446-Lab3

# EE446_Lab3_UCI_HAR_DNN_Quantization_Student_TODO (1).ipynb

7. Evaluate the Baseline Keras Model
<img width="856" height="1060" alt="image" src="https://github.com/user-attachments/assets/741e7abf-a29f-4db1-96d7-a257a1c52613" />

10. PTQ Comparison: Accuracy and Model Size
<img width="547" height="190" alt="image" src="https://github.com/user-attachments/assets/0d00ad6f-6e1a-4b86-86dc-2a470c0f195e" />

Confusion Matrix for the PTQ Int8 Model
<img width="858" height="1046" alt="image" src="https://github.com/user-attachments/assets/faf1db1a-0b34-4162-a4c3-91d97259b361" />

12. PTQ Int8 vs QAT Int8
<img width="377" height="115" alt="image" src="https://github.com/user-attachments/assets/5f5d8519-9295-400c-b55b-b370bab390e5" />

<img width="842" height="707" alt="image" src="https://github.com/user-attachments/assets/1b5d3614-8281-4fb4-81ce-ca5ea16ce21b" />

**Summary Questions:**
Which quantization method gave the smallest model size?

PTQ INT8 (and also the QAT version) dynamic range was very close though.


Which quantization method gave the best accuracy among the TensorFlow Lite models?

All but INT8.


Did QAT improve the final int8 model compared with PTQ int8?

Yes by 1.22 %.


Why is this dataset a good fit for a DNN-based TinyML workflow?

Small input size and not a crazy data type like an image or audio file. Also there was only 6 classes and the model was already pretty small before quantization.


If you were deploying this model on a resource-constrained device, which version would you choose and why?

QAT INT8 model because its the smallest and has the best accuracy, even higher than the others which is a little strange... I think integers are also better for edge devices anyway.



# EE446_UCI_HAR_DNN_Pruning_Quantization_Student_TODO (1).ipynb

Baseline Model:
<img width="827" height="1053" alt="image" src="https://github.com/user-attachments/assets/a21b3cf9-0fd5-4a0c-be9f-a10fb0141483" />

11. Part I Comparison: Accuracy and Model Size
<img width="635" height="159" alt="image" src="https://github.com/user-attachments/assets/9f8fa871-398b-4af9-a48d-fc9aed3a3901" />

Confusion Matrix for the Stripped Sparse Model
<img width="827" height="1025" alt="image" src="https://github.com/user-attachments/assets/c527fd74-57d3-4cec-b071-89ad0a938de5" />

12. Part II Comparison: Accuracy and Model Size
<img width="651" height="192" alt="image" src="https://github.com/user-attachments/assets/4cc9316a-76d0-4463-aca3-24d913a8f1dd" />

Confusion Matrix for the Stripped Sparse + Float16 Model
<img width="833" height="1014" alt="image" src="https://github.com/user-attachments/assets/8adb8180-1d01-4c49-b98a-c05e27986af0" />

**13. Summary Questions**
Write short answers to the following:

Did pruning alone reduce the TensorFlow Lite file size when the pruning wrappers were still attached?

No. The pruned model with wrappers was 1454.18 KB, which was much larger than the baseline model at 726.72 KB because the pruning masks and wrapper information were stored.


Why does strip_pruning(...) matter before export?

strip_pruning(...) removes the extra pruning wrapper layers and keeps only the learned sparse weights. This reduced the file size from 1454.18 KB to 338.85 KB.


Which model had the smallest file size in this notebook?

The pruned stripped sparse Float16 TensorFlow Lite model was the smallest at 223.71 KB.


Did float16 quantization noticeably change the test accuracy?

No. All Part II models had the same test accuracy of 0.9277, so float16 reduced size without hurting accuracy.


If you were deploying this model on a resource-constrained device, which version would you choose and why?

I would choose the pruned stripped sparse Float16 model because it had the smallest size (223.71 KB) while keeping the same 92.77% test accuracy as the other pruned models.



# EE446_UCI_HAR_DNN_Knowledge_Distillation_Pruning_Quantization_Student_TODO (1).ipynb

12. Evaluate the Distilled Student
<img width="773" height="1036" alt="image" src="https://github.com/user-attachments/assets/5a856fa2-3e4d-459a-ba4c-fa8f57072cb9" />

13. Part I Comparison: Teacher vs Student vs Distilled Student
<img width="350" height="141" alt="image" src="https://github.com/user-attachments/assets/3220e916-d2b8-45af-a0f7-f7139e1832dc" />

19. Part II Comparison: Distillation, Pruning, and Quantization
<img width="600" height="164" alt="image" src="https://github.com/user-attachments/assets/3f0bc82b-0ed9-430b-a642-feb07640a31f" />

Confusion Matrix for the Final Sparse INT8 Distilled Student
<img width="762" height="745" alt="image" src="https://github.com/user-attachments/assets/d4395f5e-cb15-46ff-8dc5-40528b6ef3b0" />

**20. Summary Questions**
Answer the following in your lab report:

How did the baseline student compare with the distilled student?

The baseline student and distilled student used the same small architecture with 80,582 parameters, but the distilled student performed better. The baseline student achieved 93.59% test accuracy, while the distilled student reached 94.06%, showing slightly improved performance without increasing model size.


Did knowledge distillation help the smaller model retain performance?

Yes. Knowledge distillation improved the small student model by transferring information from the teacher model’s soft outputs. Even though the student remained lightweight, it outperformed the baseline student and also exceeded the teacher model’s 91.99% test accuracy. I think because the teacher may have overfit to the data but was still able to teach certain learned features.


What happened to the model size after pruning and after INT8 quantization?

Pruning with masks attached initially increased file size from 316.9 KB to 634.7 KB because pruning metadata was stored. After stripping the pruning wrappers and applying sparse optimization, the size dropped to 128.0 KB. Applying INT8 quantization reduced it further to 65.5 KB, the smallest model.


Which model would you choose for Arduino deployment, and why?

I would choose the stripped sparse INT8 distilled student model because it had the smallest size (65.5 KB) while maintaining strong accuracy (93.65%). This makes it ideal for devices with limited flash memory and RAM.


Why is the final sparse INT8 model a good TinyML deployment candidate?

The final sparse INT8 model combines several TinyML advantages: low storage requirements, efficient integer arithmetic, reduced memory bandwidth, and strong predictive accuracy. It preserves nearly all of the original student model’s performance while being small enough for embedded microcontrollers.
