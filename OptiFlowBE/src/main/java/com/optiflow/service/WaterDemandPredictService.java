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
import com.optiflow.domain.Reservoir;
import com.optiflow.domain.WaterDemandPredict;
import com.optiflow.dto.WaterDemandPredictRequestDto;
import com.optiflow.dto.WaterDemandPredictResponseDto;
import com.optiflow.dto.WaterDemandPredictionItemDto;
import com.optiflow.persistence.ReservoirRepository;
import com.optiflow.persistence.WaterDemandPredictRepository;

@Service
public class WaterDemandPredictService {

	@Value("${fastapi.url}")
	private String fastapiUrl;

	@Autowired
	private WaterDemandPredictRepository waterRepo;

	@Autowired
	private ReservoirRepository reservoirRepo;

	public List<WaterDemandPredict> getAllPredicts() {
		return waterRepo.findAll();
	}

	public WaterDemandPredict savePredict(String name, String modelName, String datetime,
			List<WaterDemandPredictionItemDto> prediction, List<WaterDemandPredictionItemDto> optiflow) {
		WaterDemandPredict predict = new WaterDemandPredict();
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(name);
		int reservoirId = reservoirOptional.get().getReservoirId();
		Optional<Reservoir> reservoirEntity = reservoirRepo.findById(reservoirId);

		predict.setReservoirId(reservoirEntity.get());
		predict.setDatetime(datetime);
		predict.setUsedModel(modelName);
		predict.setPrediction(prediction);
		predict.setOptiflow(optiflow);
		return waterRepo.save(predict);
	}
	
	
	public WaterDemandPredictResponseDto getPrediction(WaterDemandPredictRequestDto requestDto) {

        String datetime = requestDto.getDatetime();
        String name = requestDto.getName();
        String modelName = requestDto.getModelName();
        
        Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(name);
		int reservoirId = reservoirOptional.get().getReservoirId();
		Optional<Reservoir> reservoirEntity = reservoirRepo.findById(reservoirId);
        // DB에서 기존 예측 데이터 조회
        Optional<WaterDemandPredict> existingPrediction = waterRepo.findByDatetimeAndReservoirIdAndUsedModel(datetime,reservoirEntity.get(), modelName);

        if (existingPrediction.isPresent()) {
            // DB에 데이터가 존재하면 DB 데이터 반환
        	WaterDemandPredictResponseDto responseDto = new WaterDemandPredictResponseDto();
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

                WaterDemandPredictResponseDto responseDto = restTemplate.postForObject(
                    fastapiUrl + "/api/predict/" + modelName,
                    request,
                    WaterDemandPredictResponseDto.class
                );

                if (responseDto == null) {
                    return new WaterDemandPredictResponseDto(); 
                } else {
                    savePredict(name, modelName, datetime, responseDto.getPrediction(), responseDto.getOptiflow());
                    return responseDto;
                }

            } catch (Exception e) {
                return new WaterDemandPredictResponseDto();
            }
        }
    }

	public Map<String, List<?>> convertToResponseMap(List<WaterDemandPredictResponseDto> datas) {
		Map<String, List<?>> responseMap = new HashMap<>();

		List<Double> predictionValueList = new ArrayList<>();
		List<Double> optiflowValueList = new ArrayList<>();
		List<Double> heightValueList = new ArrayList<>();
		List<String> timeList = new ArrayList<>();

		if (!datas.isEmpty()) {
			WaterDemandPredictResponseDto data = datas.get(0);

			for (WaterDemandPredictionItemDto item : data.getPrediction()) {
				predictionValueList.add(item.getValue());
				timeList.add(item.getTime().toString());
			}

			for (WaterDemandPredictionItemDto item : data.getOptiflow()) {
				optiflowValueList.add(item.getValue());
				heightValueList.add(item.getHeight());
			}
		}

		responseMap.put("prediction", predictionValueList);
		responseMap.put("optiflow", optiflowValueList);
		responseMap.put("time", timeList);
		responseMap.put("height", heightValueList);

		return responseMap;
	}
}
//	public WaterDemandPredictResponseDto getPrediction(WaterDemandPredictRequestDto requestDto) {
//
//		String datetime = requestDto.getDatetime();
//		String name = requestDto.getName();
//		String modelName = requestDto.getModelName();
//
//		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(name);
//		int reservoirId = reservoirOptional.get().getReservoirId();
//		Optional<Reservoir> reservoirEntity = reservoirRepo.findById(reservoirId);
//		// DB에서 기존 예측 데이터 조회
//		Optional<WaterDemandPredict> existingPrediction = waterRepo.findByDatetimeAndReservoirIdAndUsedModel(datetime,
//				reservoirEntity.get(), modelName);
//
//		if (existingPrediction.isPresent()) {
//			// DB에 데이터가 존재하면 DB 데이터 반환
//			WaterDemandPredictResponseDto responseDto = new WaterDemandPredictResponseDto();
//			responseDto.setPrediction(existingPrediction.get().getPrediction());
//			responseDto.setOptiflow(existingPrediction.get().getOptiflow());
//			return responseDto;
//		} else {
//			// DB에 데이터가 없으면 FastAPI 호출 및 DB 저장
//			RestTemplate restTemplate = new RestTemplate();
//			HttpHeaders headers = new HttpHeaders();
//			headers.setContentType(MediaType.APPLICATION_JSON);
//
//			try {
//				String requestJson = new ObjectMapper().writeValueAsString(requestDto);
//				HttpEntity<String> request = new HttpEntity<>(requestJson, headers);
//
//				WaterDemandPredictResponseDto responseDto = restTemplate.postForObject(
//						fastapiUrl + "/api/predict/" + modelName, request, WaterDemandPredictResponseDto.class);
//
//				if (responseDto == null) {
//					return new WaterDemandPredictResponseDto();
//				} else {
//					savePredict(name, modelName, datetime, responseDto.getPrediction(), responseDto.getOptiflow());
//					return responseDto;
//				}
//
//			} catch (Exception e) {
//				return new WaterDemandPredictResponseDto();
//			}
//		}
//	}



//public WaterDemandPredictResponseDto getPrediction(WaterDemandPredictRequestDto requestDto) {
//
//    String datetime = requestDto.getDatetime();
//    String name = requestDto.getName();
//    String modelName = requestDto.getModelName();
//
//    Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(name);
//    int reservoirId = reservoirOptional.get().getReservoirId();
//    Optional<Reservoir> reservoirEntity = reservoirRepo.findById(reservoirId);
//    // DB에서 기존 예측 데이터 조회
//    Optional<WaterDemandPredict> existingPrediction = waterRepo.findByDatetimeAndReservoirIdAndUsedModel(datetime,
//            reservoirEntity.get(), modelName);
//
//    if (existingPrediction.isPresent()) {
//        // DB에 데이터가 존재하면 DB 데이터 반환
//        WaterDemandPredictResponseDto responseDto = new WaterDemandPredictResponseDto();
//        responseDto.setPrediction(existingPrediction.get().getPrediction());
//        responseDto.setOptiflow(existingPrediction.get().getOptiflow());
//        return responseDto;
//    } else {
//        // DB에 데이터가 없으면 FastAPI 호출 및 DB 저장
//        RestTemplate restTemplate = new RestTemplate(); // 필요하다면 빈 주입 방식으로 변경 고려
//        // HttpHeaders headers = new HttpHeaders(); // Content-Type은 기본적으로 JSON으로 설정됨 (RestTemplate 기본 설정 확인)
//        // headers.setContentType(MediaType.APPLICATION_JSON); // 불필요할 수 있음
//
//        try {
//            // ObjectMapper objectMapper = new ObjectMapper(); // 재사용하도록 빈 등록 또는 static 변수 사용 고려
//            // String requestJson = objectMapper.writeValueAsString(requestDto); // 더 이상 JSON 문자열로 변환할 필요 없음
//            // HttpEntity<String> request = new HttpEntity<>(requestJson, headers); // 불필요
//
//            // **수정: postForObject의 두 번째 인자에 requestDto 객체를 직접 전달, URL 수정**
//            WaterDemandPredictResponseDto responseDto = restTemplate.postForObject(
//                    fastapiUrl + "/api/predict/" + modelName, // URL 수정: request 객체 제거, modelName은 경로 파라미터로 유지
//                    requestDto,             // 요청 본문: requestDto 객체 직접 전달
//                    WaterDemandPredictResponseDto.class);
//
//            // responseDto == null 체크 불필요
//            savePredict(name, modelName, datetime, responseDto.getPrediction(), responseDto.getOptiflow());
//            return responseDto;
//
//
//        } catch (RestClientException e) { // RestTemplate 관련 예외 구체적으로 처리
//            System.err.println("API 호출 오류 발생: " + e.getMessage()); // 로그 출력 또는 로깅 라이브러리 사용
//            e.printStackTrace(); // 스택 트레이스 출력 (디버깅 용이)
//            // return new WaterDemandPredictResponseDto(); // 빈 DTO 반환 대신 null 또는 예외 다시 던지기 고려
//            return null; // 예시: null 반환
//        }
//    }
//}

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
