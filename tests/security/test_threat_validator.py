"""Tests for Security/Threat Validator module."""

import pytest
from unittest.mock import Mock

from sage.security.threat_validator import (
    ThreatValidator,
    ThreatDefinition,
    ThreatResult,
    AttackTreeNode,
    STRIDEReport,
    STRIDECategory,
    ThreatSeverity,
    MitigationStatus,
    MCP_THREATS,
    A2A_THREATS,
    ALL_THREATS,
    quick_security_assessment,
    get_threat_by_id,
    get_stride_mcp_table,
)


class TestThreatValidator:
    """Tests for ThreatValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a threat validator."""
        return ThreatValidator(safe_mode=True)

    def test_initialization(self):
        """Test validator initialization."""
        validator = ThreatValidator()
        assert validator.safe_mode is True
        assert validator.mcp_client is None
        assert validator.a2a_client is None

    def test_initialization_with_clients(self):
        """Test initialization with clients."""
        mock_mcp = Mock()
        mock_a2a = Mock()

        validator = ThreatValidator(mcp_client=mock_mcp, a2a_client=mock_a2a)

        assert validator.mcp_client == mock_mcp
        assert validator.a2a_client == mock_a2a

    def test_test_server_spoofing(self, validator):
        """Test server spoofing test."""
        result = validator.test_server_spoofing("https://test.example.com")

        assert isinstance(result, ThreatResult)
        assert result.threat.threat_id == "MCP-S01"
        assert len(result.evidence) > 0

    def test_test_server_spoofing_http(self, validator):
        """Test spoofing detection for HTTP."""
        result = validator.test_server_spoofing("http://insecure.example.com")

        assert result.vulnerable
        assert "HTTP" in " ".join(result.evidence)

    def test_test_prompt_injection(self, validator):
        """Test prompt injection test."""
        result = validator.test_prompt_injection_via_tool()

        assert isinstance(result, ThreatResult)
        assert result.threat.threat_id == "MCP-I02"
        assert "injection_rate" in result.details

    def test_test_prompt_injection_custom_payloads(self, validator):
        """Test with custom injection payloads."""
        payloads = ["Custom payload 1", "Custom payload 2"]

        result = validator.test_prompt_injection_via_tool(injection_payloads=payloads)

        assert result.details["payloads_tested"] == 2

    def test_test_capability_probing(self, validator):
        """Test capability probing test."""
        result = validator.test_capability_probing()

        assert isinstance(result, ThreatResult)
        assert result.threat.threat_id == "MCP-E01"

    def test_test_context_leakage(self, validator):
        """Test context leakage test."""
        result = validator.test_context_leakage()

        assert isinstance(result, ThreatResult)
        assert result.threat.threat_id == "MCP-I01"

    def test_test_context_leakage_with_data(self, validator):
        """Test with sample sensitive data."""
        sample_data = {"email": "test@example.com", "ssn": "123-45-6789"}

        result = validator.test_context_leakage(sample_context=sample_data)
        assert len(result.evidence) > 0

    def test_test_cross_agent_escalation(self, validator):
        """Test cross-agent escalation test."""
        result = validator.test_cross_agent_escalation()

        assert isinstance(result, ThreatResult)
        assert result.threat.threat_id == "A2A-E01"

    def test_test_cross_agent_escalation_custom_caps(self, validator):
        """Test with custom capabilities."""
        capabilities = {"agent_1": ["read"], "agent_2": ["write"], "agent_3": ["admin"]}

        result = validator.test_cross_agent_escalation(agent_capabilities=capabilities)

        assert "agent_1" in str(result.details)

    def test_build_attack_tree(self, validator):
        """Test attack tree generation."""
        tree = validator.build_attack_tree()

        assert isinstance(tree, AttackTreeNode)
        assert tree.node_type == "OR"
        assert len(tree.children) == 3  # 3 attack paths

    def test_build_attack_tree_custom_goal(self, validator):
        """Test with custom goal."""
        tree = validator.build_attack_tree(goal="Data Exfiltration")

        assert tree.description == "Data Exfiltration"

    def test_generate_stride_report(self, validator):
        """Test STRIDE report generation."""
        report = validator.generate_stride_report()

        assert isinstance(report, STRIDEReport)
        assert len(report.results) > 0
        assert len(report.attack_trees) > 0
        assert len(report.recommendations) >= 0

    def test_generate_stride_report_markdown(self, validator):
        """Test markdown report generation."""
        report = validator.generate_stride_report()
        markdown = report.to_markdown()

        assert "# STRIDE Security Analysis Report" in markdown
        assert "Target" in markdown

    def test_stride_report_to_dict(self, validator):
        """Test report serialization."""
        report = validator.generate_stride_report()
        d = report.to_dict()

        assert "target" in d
        assert "vulnerability_count" in d
        assert "results" in d


class TestThreatDefinitions:
    """Tests for threat definitions."""

    def test_mcp_threats_exist(self):
        """Test MCP threats are defined."""
        assert len(MCP_THREATS) > 0

    def test_a2a_threats_exist(self):
        """Test A2A threats are defined."""
        assert len(A2A_THREATS) > 0

    def test_all_threats_combined(self):
        """Test all threats combines both."""
        assert len(ALL_THREATS) == len(MCP_THREATS) + len(A2A_THREATS)

    def test_threat_has_required_fields(self):
        """Test threats have required fields."""
        for threat in ALL_THREATS:
            assert threat.threat_id
            assert threat.category in STRIDECategory
            assert threat.name
            assert threat.description
            assert threat.severity in ThreatSeverity

    def test_threat_ids_unique(self):
        """Test threat IDs are unique."""
        ids = [t.threat_id for t in ALL_THREATS]
        assert len(ids) == len(set(ids))

    def test_all_stride_categories_covered(self):
        """Test all STRIDE categories have threats."""
        categories = set()
        for threat in ALL_THREATS:
            categories.add(threat.category)

        assert STRIDECategory.SPOOFING in categories
        assert STRIDECategory.TAMPERING in categories
        assert STRIDECategory.INFORMATION_DISCLOSURE in categories


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_quick_security_assessment(self):
        """Test quick assessment function."""
        report = quick_security_assessment(safe_mode=True)

        assert isinstance(report, STRIDEReport)
        assert report.target == "test_environment"

    def test_get_threat_by_id_found(self):
        """Test getting threat by ID."""
        threat = get_threat_by_id("MCP-S01")

        assert threat is not None
        assert threat.threat_id == "MCP-S01"

    def test_get_threat_by_id_not_found(self):
        """Test getting non-existent threat."""
        threat = get_threat_by_id("INVALID-ID")

        assert threat is None

    def test_get_stride_mcp_table(self):
        """Test STRIDE MCP table."""
        table = get_stride_mcp_table()

        assert "Spoofing" in table
        assert "Tampering" in table
        assert "threat" in table["Spoofing"]
        assert "mitigation_status" in table["Spoofing"]


class TestEnums:
    """Tests for enums."""

    def test_stride_category_values(self):
        """Test STRIDE category values."""
        assert STRIDECategory.SPOOFING.value == "spoofing"
        assert STRIDECategory.TAMPERING.value == "tampering"
        assert STRIDECategory.REPUDIATION.value == "repudiation"
        assert STRIDECategory.INFORMATION_DISCLOSURE.value == "information_disclosure"
        assert STRIDECategory.DENIAL_OF_SERVICE.value == "denial_of_service"
        assert STRIDECategory.ELEVATION_OF_PRIVILEGE.value == "elevation_of_privilege"

    def test_threat_severity_ordering(self):
        """Test severity levels are ordered."""
        assert ThreatSeverity.LOW.value < ThreatSeverity.MEDIUM.value
        assert ThreatSeverity.MEDIUM.value < ThreatSeverity.HIGH.value
        assert ThreatSeverity.HIGH.value < ThreatSeverity.CRITICAL.value

    def test_mitigation_status_values(self):
        """Test mitigation status values."""
        assert MitigationStatus.NONE.value == "none"
        assert MitigationStatus.PARTIAL.value == "partial"
        assert MitigationStatus.FULL.value == "full"


class TestThreatResult:
    """Tests for ThreatResult dataclass."""

    def test_threat_result_creation(self):
        """Test ThreatResult creation."""
        threat = MCP_THREATS[0]
        result = ThreatResult(
            threat=threat,
            vulnerable=True,
            mitigation_status=MitigationStatus.NONE,
            evidence=["Test evidence"],
            test_method="unit_test",
            recommendations=["Fix it"],
        )

        assert result.vulnerable
        assert result.mitigation_status == MitigationStatus.NONE

    def test_threat_result_to_dict(self):
        """Test ThreatResult serialization."""
        threat = MCP_THREATS[0]
        result = ThreatResult(
            threat=threat,
            vulnerable=False,
            mitigation_status=MitigationStatus.FULL,
            evidence=[],
            test_method="test",
            recommendations=[],
        )

        d = result.to_dict()
        assert d["threat_id"] == threat.threat_id
        assert d["vulnerable"] is False


class TestAttackTreeNode:
    """Tests for AttackTreeNode dataclass."""

    def test_attack_tree_node_creation(self):
        """Test AttackTreeNode creation."""
        node = AttackTreeNode(
            node_id="root",
            description="Test attack",
            node_type="OR",
            difficulty="medium",
        )

        assert node.node_id == "root"
        assert node.node_type == "OR"

    def test_attack_tree_with_children(self):
        """Test AttackTreeNode with children."""
        child1 = AttackTreeNode(
            node_id="c1", description="Child 1", node_type="LEAF", difficulty="easy"
        )
        child2 = AttackTreeNode(
            node_id="c2", description="Child 2", node_type="LEAF", difficulty="hard"
        )

        root = AttackTreeNode(
            node_id="root",
            description="Root",
            node_type="AND",
            difficulty="medium",
            children=[child1, child2],
        )

        assert len(root.children) == 2

    def test_attack_tree_to_dict(self):
        """Test AttackTreeNode serialization."""
        node = AttackTreeNode(
            node_id="test",
            description="Test",
            node_type="LEAF",
            difficulty="easy",
            mitigations=["Mitigation 1"],
        )

        d = node.to_dict()
        assert d["node_id"] == "test"
        assert d["mitigations"] == ["Mitigation 1"]
