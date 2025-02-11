package com.optiflow.service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.optiflow.domain.Predict;
import com.optiflow.domain.Reservoir;
import com.optiflow.dto.PredictRequestDto;
import com.optiflow.dto.PredictResponseDto;
import com.optiflow.dto.PredictionItemDto;
import com.optiflow.persistence.PredictRepository;
import com.optiflow.persistence.ReservoirRepository;

@Service
public class PredictService {

	@Value("${fastapi.url}")
	private String fastapiUrl;
	
	@Autowired
	private PredictRepository predictRepo;
	
	@Autowired
	private ReservoirRepository reservoirRepo;

	public List<Predict> getAllPredicts() {
		return predictRepo.findAll();
	}
	
	public Predict savePredict(String name, String datetime, List<PredictionItemDto> prediction, List<PredictionItemDto> optiflow) {
		Predict predict = new Predict();
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(name);
		int reservoirId = reservoirOptional.get().getReservoirId();
		Optional<Reservoir> reservoirEntity = reservoirRepo.findById(reservoirId);
		
		// Reservoir 테이블의 기본 키(primary key) 인 reservoirId 값이 외래 키(foreign key) 로 저장
		predict.setReservoirId(reservoirEntity.get());
		predict.setDatetime(datetime);
		predict.setPrediction(prediction);
		predict.setOptiflow(optiflow);
		return predictRepo.save(predict);
	}
	
	public PredictResponseDto getPrediction(PredictRequestDto requestDto) {

        String datetime = requestDto.getDatetime();
        String name = requestDto.getName();
        // DB에서 datetime으로 기존 예측 데이터 조회
        Optional<Predict> existingPrediction = predictRepo.findByDatetime(datetime);

        if (existingPrediction.isPresent()) {
            // DB에 데이터가 존재하면 DB 데이터 반환
            PredictResponseDto responseDto = new PredictResponseDto();
            responseDto.setPrediction(existingPrediction.get().getPrediction());
            responseDto.setOptiflow(existingPrediction.get().getOptiflow());
            return responseDto;
        } else {
            // DB에 데이터가 없으면 FastAPI 호출 및 DB 저장
            RestTemplate restTemplate = new RestTemplate();
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            try {
                String requestJson = new ObjectMapper().writeValueAsString(requestDto);
                HttpEntity<String> request = new HttpEntity<>(requestJson, headers);

                PredictResponseDto responseDto = restTemplate.postForObject(
                    fastapiUrl + "/api/predict",
                    request,
                    PredictResponseDto.class
                );

                if (responseDto == null) {
                    return new PredictResponseDto(); // 빈 결과 반환
                } else {
                    savePredict(name, datetime, responseDto.getPrediction(), responseDto.getOptiflow());
                    return responseDto;
                }

            } catch (Exception e) {
                return new PredictResponseDto(); // 예외 발생 시 빈 결과 반환
            }
        }
    }
	
	public Map<String, List<?>> convertToResponseMap(List<PredictResponseDto> datas) {
	    Map<String, List<?>> responseMap = new HashMap<>();

	    List<Double> predictionValueList = new ArrayList<>();
	    List<Double> optiflowValueList = new ArrayList<>();
	    List<String> timeList = new ArrayList<>();

	    if (!datas.isEmpty() && datas.get(0) != null) { 
	        PredictResponseDto data = datas.get(0);

	        if (data.getPrediction() != null && !data.getPrediction().isEmpty()) {
	        	
	            for (PredictionItemDto item : data.getPrediction()) {
	                if (item != null) {
	                    predictionValueList.add(item.getValue());
	                }
	            }
	            
                for (PredictionItemDto item : data.getPrediction()) {
                    if (item != null && item.getTime() != null) {
                        timeList.add(item.getTime().toString());
                    }
                }
	        }
	        
	        if (data.getOptiflow() != null && !data.getOptiflow().isEmpty()) {
	            for (PredictionItemDto item : data.getOptiflow()) {
	                if (item != null) {
	                    optiflowValueList.add(item.getValue());
	                }
	            }
	        }
	    }
	    
        responseMap.put("prediction", predictionValueList);
        responseMap.put("optiflow", optiflowValueList);
	    responseMap.put("time", timeList);

	    return responseMap;
	}
}
//	public PredictResponseDto getPrediction(PredictRequestDto requestDto) {
//	    log.info("PredictService.getPrediction called with request: {}", requestDto);
//
//	    RestTemplate restTemplate = new RestTemplate();
//	    HttpHeaders headers = new HttpHeaders();
//	    headers.setContentType(MediaType.APPLICATION_JSON);
//
//	    try {
//	        // Jackson ObjectMapper를 활용해 JSON 문자열로 변환 후 전송
//	        String requestJson = new ObjectMapper().writeValueAsString(requestDto);
//	        HttpEntity<String> request = new HttpEntity<>(requestJson, headers);
//	        
//	        PredictResponseDto responseDto = restTemplate.postForObject(
//	            fastapiUrl + "/api/predict", 
//	            request, 
//	            PredictResponseDto.class
//	        );
//	        
//	        if (responseDto == null) {
//	            log.error("FastAPI prediction failed or returned null response.");
//	            return new PredictResponseDto();  // 빈 결과 반환
//	        } else {
//	            log.info("FastAPI prediction successful. Response: {}", responseDto);
//	            savePredict(requestDto.getDatetime(), responseDto.getPrediction());
//	            return responseDto;
//	        }
//
//	    } catch (Exception e) {
//	        log.error("Error while calling FastAPI for prediction: ", e);
//	        return new PredictResponseDto();  // 예외 발생 시 빈 결과 반환
//	    }
//	}
//	public PredictResponseDto getPrediction(PredictRequestDto requestDto) {
//	    log.info("PredictService.getPrediction called with request: {}", requestDto);
//
//	    RestTemplate restTemplate = new RestTemplate();
//	    HttpHeaders headers = new HttpHeaders();
//	    headers.setContentType(MediaType.APPLICATION_JSON);
//
//	    HttpEntity<PredictRequestDto> request = new HttpEntity<>(requestDto, headers);
//	    
//	    try {
//	        ResponseEntity<PredictResponseDto> response = restTemplate.exchange(
//	            fastapiUrl + "/api/predict",
//	            HttpMethod.POST,
//	            request,
//	            PredictResponseDto.class
//	        );
//
//	        if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
//	            PredictResponseDto responseDto = response.getBody();
//	            log.info("FastAPI prediction successful. Response: {}", responseDto);
//	            savePredict(requestDto.getDatetime(), responseDto.getPrediction());
//	            return responseDto;
//	        } else {
//	            log.error("FastAPI prediction failed with status: {}", response.getStatusCode());
//	            return new PredictResponseDto();  // 빈 결과 반환
//	        }
//	    } catch (Exception e) {
//	        log.error("Error while calling FastAPI for prediction: ", e);
//	        return new PredictResponseDto();  // 예외 발생 시 빈 결과 반환
//	    }
//	}


//	public PredictResponseDto getPrediction(PredictRequestDto requestDto) {
//		log.info("PredictService.getPrediction called with request: {}", requestDto);
//
//		RestTemplate restTemplate = new RestTemplate();
//		HttpHeaders headers = new HttpHeaders();
//		headers.setContentType(MediaType.APPLICATION_JSON);
//
//		HttpEntity<PredictRequestDto> request = new HttpEntity<>(requestDto, headers);
//		PredictResponseDto responseDto = new PredictResponseDto();
//
//		try {
//			responseDto = restTemplate.postForObject(fastapiUrl + "/api/predict", request, PredictResponseDto.class); 
//			
//			if (responseDto == null) {
//				log.error("FastAPI prediction failed or returned null response: {}", responseDto);
//				responseDto = new PredictResponseDto();
//			} else {
//				log.info("FastAPI prediction successful. Response: {}", responseDto); // 성공 응답 로깅
//				savePredict(requestDto.getDatetime(), responseDto.getPrediction());
//			}
//		} catch (Exception e) {
//			log.error("Error while calling FastAPI for prediction: ", e);
//			responseDto = new PredictResponseDto();
//		}
//		return responseDto;
//	}

//	private final RestTemplate restTemplate; // Bean 주입된 RestTemplate 사용
//
//	public PredictService(RestTemplate restTemplate) {
//		this.restTemplate = restTemplate;
//	}
//
//	private final ObjectMapper objectMapper = new ObjectMapper(); // ObjectMapper Bean 등록 후 주입받아도 좋음
//	private void savePredictList(String datetime, List<PredictResponseDto> predictionList) {
//        log.info("Saving {} predictions for datetime: {}", predictionList.size(), datetime);
//        for (PredictResponseDto prediction : predictionList) {
//            savePredict(datetime, prediction.getResult());
//        }
//    }
