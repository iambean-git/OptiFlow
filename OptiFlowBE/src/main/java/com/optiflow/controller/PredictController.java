package com.optiflow.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;

import com.optiflow.domain.Predict;
import com.optiflow.service.PredictService;

@Controller
@RequestMapping("/api")
public class PredictController {

	@Autowired
	private PredictService predictService;

	@PostMapping("/save")
    public ResponseEntity<Predict> savePredict(@RequestBody Predict predict) {
		System.out.println("Received from fastAPI: " + predict);
        // 받은 모델 데이터를 DB에 저장
		Predict savedPredict = predictService.savePredict(predict.getText(), predict.getResult());
        return ResponseEntity.ok(savedPredict);
    }
	
	@GetMapping("/results")
	public ResponseEntity<List<Predict>> getAllPredicts(){
		List<Predict> predictList = predictService.getAllPredicts();
		return ResponseEntity.ok(predictList);
	}
}
