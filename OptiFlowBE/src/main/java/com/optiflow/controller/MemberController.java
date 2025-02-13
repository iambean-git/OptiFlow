package com.optiflow.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.Member;
import com.optiflow.service.MemberService;

import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/api/members")
@Tag(name = "Member API", description = "멤버 관련 API")
public class MemberController {
	
	@Autowired
	private MemberService memberService;
	
    @PostMapping
    public ResponseEntity<Member> createOrUpdateMember(@RequestBody Member member) {
        Member savedMember = memberService.saveMember(member);
        return ResponseEntity.ok(savedMember);
    }

    @GetMapping
    public ResponseEntity<List<Member>> getAllMembers() {
        List<Member> members = memberService.getAllMembers();
        return ResponseEntity.ok(members);
    }

    @GetMapping("/{username}")
    public ResponseEntity<Member> getMemberByUsername(@PathVariable String username) {
        return memberService.getMemberByUsername(username)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{username}")
    public ResponseEntity<Void> deleteMember(@PathVariable String username) {
        memberService.deleteMember(username);
        return ResponseEntity.noContent().build();
    }

}
