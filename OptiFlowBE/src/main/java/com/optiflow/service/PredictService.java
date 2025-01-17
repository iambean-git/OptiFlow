package com.optiflow.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.optiflow.domain.Predict;
import com.optiflow.persistence.PredictRepository;

@Service
public class PredictService {
	
	@Autowired
	private PredictRepository predictRepo;
	
	public Predict savePredict(String text, String result) {
		Predict predict = new Predict();
		predict.setText(text);
		predict.setResult(result);	
		return predictRepo.save(predict);
	}
	
	public List<Predict> getAllPredicts(){
		return predictRepo.findAll();
	}
}
