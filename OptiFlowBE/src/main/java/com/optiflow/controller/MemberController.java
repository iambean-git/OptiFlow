package com.optiflow.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;

import com.optiflow.domain.Member;
import com.optiflow.service.MemberService;

@Controller
@RequestMapping("/api/members")
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
