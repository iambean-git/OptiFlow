package com.optiflow.controller;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
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
	
	private static final Logger log = LoggerFactory.getLogger(PredictController.class); // Logger 추가
	
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
	
	@PostMapping("/predict") // 예측 요청 API 엔드포인트 추가
    public ResponseEntity<PredictResponseDto> getPrediction(@RequestBody PredictRequestDto requestDto) {
        log.info("Received prediction request: {}", requestDto); // 예측 요청 로깅
        PredictResponseDto responseDto = predictService.getPrediction(requestDto);
        return ResponseEntity.ok(responseDto); // 예측 결과 응답 DTO 반환
    }
//	@PostMapping("/predict") // 예측 요청 API 엔드포인트 추가
//    public ResponseEntity<PredictResponseDto> getPrediction(@RequestBody PredictRequestDto requestDto) {
//        log.info("Received prediction request: {}", requestDto); // 예측 요청 로깅
//        PredictResponseDto responseDto = predictService.getPrediction(requestDto);
//        return ResponseEntity.ok(responseDto); // 예측 결과 응답 DTO 반환
//    }
}
