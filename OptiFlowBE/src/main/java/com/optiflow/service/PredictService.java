package com.optiflow.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.optiflow.domain.Predict;
import com.optiflow.dto.PredictRequestDto;
import com.optiflow.dto.PredictResponseDto;
import com.optiflow.persistence.PredictRepository;

import java.util.List;

@Service
public class PredictService {
	
	private static final Logger log = LoggerFactory.getLogger(PredictService.class); // Logger 추가

    @Value("${fastapi.url}")
    private String fastapiUrl;
    
	@Autowired
	private PredictRepository predictRepo;
	
	public Predict savePredict(String text, List<String> result) {
		Predict predict = new Predict();
		predict.setText(text);
		predict.setResult(result);	
		return predictRepo.save(predict);
	}
	
	public List<Predict> getAllPredicts(){
		return predictRepo.findAll();
	}
	
	public PredictResponseDto getPrediction(PredictRequestDto requestDto) {
        log.info("PredictService.getPrediction called with request: {}", requestDto);

        RestTemplate restTemplate = new RestTemplate();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<PredictRequestDto> request = new HttpEntity<>(requestDto, headers);
        PredictResponseDto responseDto = new PredictResponseDto();

        try {
            responseDto = restTemplate.postForObject(fastapiUrl + "/predict", request, PredictResponseDto.class); // FastAPI "/predict" 엔드포인트 호출
            if (responseDto == null) {
                log.error("FastAPI prediction failed or returned null response: {}", responseDto);
                responseDto = new PredictResponseDto();
            } else {
                log.info("FastAPI prediction successful. Response: {}", responseDto); // 성공 응답 로깅
                savePredict(requestDto.getDate(), responseDto.getResult());
            }
        } catch (Exception e) { 
            log.error("Error while calling FastAPI for prediction: ", e);
            responseDto = new PredictResponseDto();
        }
        return responseDto;
    }
}
