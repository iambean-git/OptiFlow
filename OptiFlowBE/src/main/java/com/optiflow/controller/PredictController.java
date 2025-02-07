package com.optiflow.controller;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.Predict;
import com.optiflow.dto.PredictRequestDto;
import com.optiflow.dto.PredictResponseDto;
import com.optiflow.service.PredictService;

@RestController
@RequestMapping("/api")
public class PredictController {
	
	private static final Logger log = LoggerFactory.getLogger(PredictController.class);
	
	@Autowired
	private PredictService predictService;

	@PostMapping("/save")
    public ResponseEntity<Predict> savePredict(@RequestBody Predict predict) {
		System.out.println("Received from fastAPI: " + predict);
        // 받은 모델 데이터를 DB에 저장
		Predict savedPredict = predictService.savePredict(predict.getDatetime(), predict.getResult());
        return ResponseEntity.ok(savedPredict);
    }
	
	@GetMapping("/results")
	public ResponseEntity<List<Predict>> getAllPredicts(){
		List<Predict> predictList = predictService.getAllPredicts();
		return ResponseEntity.ok(predictList);
	}
	
	@GetMapping("/predict/{datetime}")
    public ResponseEntity<PredictResponseDto> getPrediction(@PathVariable String datetime) {
		log.info("Received prediction request with datetime: {}", datetime);
        PredictRequestDto requestDto = new PredictRequestDto(); 
        requestDto.setDatetime(datetime); // PathVariable 로 받은 datetime 값을 DTO 에 설정
        PredictResponseDto responseDto = predictService.getPrediction(requestDto);
        return ResponseEntity.ok(responseDto); // 예측 결과 응답 DTO 반환
    }
}
