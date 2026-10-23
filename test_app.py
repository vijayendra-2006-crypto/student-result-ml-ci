
import unittest
from app import app


class TestStudentResultAPI(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")

    def test_pass_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 95,
                "internal_marks": 90,
                "assignment_marks": 95,
                "previous_score": 90
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["prediction"], "PASS")

    def test_fail_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 30,
                "internal_marks": 20,
                "assignment_marks": 25,
                "previous_score": 20
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["prediction"], "FAIL")

    def test_missing_field(self):
        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85,
                "assignment_marks": 90
            }
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
