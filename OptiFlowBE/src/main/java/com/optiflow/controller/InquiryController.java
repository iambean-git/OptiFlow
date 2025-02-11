package com.optiflow.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.Inquiry;
import com.optiflow.service.InquiryService;

@RestController
@RequestMapping("/api/inquiries")
public class InquiryController {
	
	@Autowired
    private InquiryService inquiryService;

    @PostMapping
    public ResponseEntity<Inquiry> createInquiry(@RequestBody Inquiry inquiry) {
        Inquiry savedInquiry = inquiryService.saveInquiry(inquiry);
        return new ResponseEntity<>(savedInquiry, HttpStatus.CREATED);
    }
    
    @GetMapping
    public ResponseEntity<List<Inquiry>> getAllInquirys(){
    	List<Inquiry> inquiries = inquiryService.getAllInquirys();
    	return new ResponseEntity<>(inquiries, HttpStatus.OK);
    }
    
    @GetMapping("/unconfirmed") // staff_confirmed 가 false 인 문의 목록 조회 API
    public ResponseEntity<List<Inquiry>> getUnconfirmedInquiries() {
        List<Inquiry> unconfirmedInquiries = inquiryService.getUnconfirmedInquiries();
        return new ResponseEntity<>(unconfirmedInquiries, HttpStatus.OK);
    }
    
    @GetMapping("/{inquiryId}") // 특정 문의 ID 조회 및 staff_confirmed 업데이트 API
    public ResponseEntity<Inquiry> getInquiryById(@PathVariable Long inquiryId) {
        Inquiry inquiry = inquiryService.getInquiryById(inquiryId);
        if (inquiry == null) {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }

        if (!inquiry.getStaffConfirmed()) {
            inquiry.setStaffConfirmed(true);
            inquiryService.saveInquiry(inquiry);
        }
        return new ResponseEntity<>(inquiry, HttpStatus.OK);
    }
}
