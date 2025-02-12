package com.optiflow.controller;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.Predict;
import com.optiflow.domain.Reservoir;
import com.optiflow.dto.PredictRequestDto;
import com.optiflow.dto.PredictResponseDto;
import com.optiflow.persistence.ReservoirDataRepository;
import com.optiflow.persistence.ReservoirRepository;
import com.optiflow.service.PredictService;

@RestController
@RequestMapping("/api")
public class PredictController {
	
	private static final Logger log = LoggerFactory.getLogger(PredictController.class);
	
	@Autowired
	private PredictService predictService;
	
	@Autowired
	private ReservoirRepository reservoirRepo;
	
	@Autowired
	private ReservoirDataRepository reservoirDataRepo;

	@GetMapping("/results")
	public ResponseEntity<List<Predict>> getAllPredicts(){
		List<Predict> predictList = predictService.getAllPredicts();
		return ResponseEntity.ok(predictList);
	}
	
	@GetMapping("/predict/{reservoirName}/{datetime}")
    public ResponseEntity<Map<String, List<?>>> getPrediction(@PathVariable String reservoirName, @PathVariable String datetime) {
		log.info("Received prediction request with datetime: {}", datetime);
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName);
		Float reservoirArea = reservoirOptional.get().getArea();
		LocalDateTime localDateTime = LocalDateTime.parse(datetime);
		int reservoirId = reservoirOptional.get().getReservoirId();
		Float height = reservoirDataRepo.findHeightByReservoirIdAndObservationTime(reservoirId, localDateTime);
		Float waterLevel = reservoirArea * height;
        PredictRequestDto requestDto = new PredictRequestDto(); 

        requestDto.setName(reservoirName);
        requestDto.setDatetime(datetime);
        requestDto.setWaterLevel(waterLevel);
        PredictResponseDto responseDto = predictService.getPrediction(requestDto);
        List<PredictResponseDto> datas = new ArrayList<>();
        datas.add(responseDto);

        Map<String, List<?>> responseMap = predictService.convertToResponseMap(datas);        
        return ResponseEntity.ok(responseMap);
    }

}

//	@PostMapping("/save")
//    public ResponseEntity<Predict> savePredict(@RequestBody Predict predict) {
//		System.out.println("Received from fastAPI: " + predict);
//        // 받은 모델 데이터를 DB에 저장
//		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(predict.getReservoirId());
//		int reservoirId = reservoirOptional.get().getReservoirId();
//		Predict savedPredict = predictService.savePredict(predict.getDatetime(), predict.getPrediction(), predict.getOptiflow());
//        return ResponseEntity.ok(savedPredict);
//    }