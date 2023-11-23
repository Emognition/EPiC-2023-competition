### Organizers edit
Original submission repository -> [https://github.com/shubhamcogsci/EPiC-submission](https://github.com/shubhamcogsci/EPiC-submission)


# EPiC-submission
---

Our team has four members. Shubham, Shranayak, Rushi and Arpit. In our approach, we first sampled all data to 200 Hz. We then extracted eight features from physiology data with the neurokit library. Then we employed a two-second delay window to adjust physiology and corresponding annotations data. Finally, we trained our data using a decision tree regression model and predicted results. 


There are separate code files associated with each step. The file clean_physio_data.ipynb applies the neurokit library and gives features of data. We used eight features: 
heart rate from ECG data, BVP clean data, skin conductance response from GSR data, respiration rate from RSP data, cleaned skin temperature data, and the amplitudes from three muscles data. The file merge_data.ipynb samples data at 200 Hz and then merges it. The file train_model.ipynb fits the data to regression mode. The file predict_newdata.ipynb used the model to predict new data. 



Additional Material will be found here : https://drive.google.com/drive/folders/1EV6MVkkaqTs1Dxov2bBywKSCch2_TR_Y


